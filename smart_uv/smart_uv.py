import numpy as np
from collections import defaultdict
import rectpack
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def get_face_normals_and_areas(vertices, faces):
    """Calculate the normal and area for each face."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Cross product of edges gives a vector whose magnitude is 2 * area
    # and whose direction is the normal.
    cross_prod = np.cross(v1 - v0, v2 - v0)
    
    # Magnitudes (2 * area)
    magnitudes = np.linalg.norm(cross_prod, axis=1)
    
    # Avoid division by zero
    valid_indices = magnitudes > 1e-8
    
    areas = np.zeros_like(magnitudes)
    areas[valid_indices] = magnitudes[valid_indices] * 0.5
    
    normals = np.zeros_like(cross_prod)
    normals[valid_indices] = cross_prod[valid_indices] / magnitudes[valid_indices, np.newaxis]
    
    return normals, areas

def axis_dominant_v3_to_m3(normal):
    """
    Equivalent to Blender's axis_dominant_v3_to_m3.
    Creates a 3x3 transformation matrix from a normal, mapping the
    dominant axis of the face to Z, making it suitable for 2D XY projection.
    """
    z_axis = normal.copy()
    abs_z = np.abs(z_axis)
    # The dominant axis in the normal (last axis of the array)
    dominant_idx = np.argmax(abs_z, axis=-1)
    
    # We want to handle both single vectors and arrays of vectors
    is_1d = (z_axis.ndim == 1)
    if is_1d:
        z_axis = z_axis.reshape(1, 3)
        dominant_idx = np.array([dominant_idx])
        
    num_vectors = z_axis.shape[0]
    axis_test = np.zeros((num_vectors, 3), dtype=z_axis.dtype)
    
    # Choose a perpendicular axis based on the dominant one
    axis_test[dominant_idx == 0, 1] = 1.0
    axis_test[dominant_idx == 1, 2] = 1.0
    axis_test[dominant_idx == 2, 0] = 1.0
        
    # X axis is cross of Z and the test axis (z_axis cross axis_test would invert it)
    x_axis = np.cross(axis_test, z_axis, axis=1)
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
    
    # Y axis is cross of Z and X
    y_axis = np.cross(z_axis, x_axis, axis=1)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1, keepdims=True)
    
    # Blender transposes this matrix
    mat = np.stack([x_axis, y_axis, z_axis], axis=1) # (N, 3, 3)
    
    if is_1d:
        return mat[0]
    return mat

def calculate_project_normals(normals, areas, angle_limit_half_cos, angle_limit_cos, area_weight=0.0):
    """
    Mirrors `smart_uv_project_calculate_project_normals`.
    Analyzes face normals to deduce optimal projection axes.
    """
    num_faces = len(normals)
    if num_faces == 0:
        return []
        
    project_normal_array = []
    
    # Tags to track which faces have been 'covered' (assigned to a group during grouping phase)
    tagged = np.zeros(num_faces, dtype=bool)
    
    # 1. Start with the largest face normal
    sorted_face_indices = np.argsort(areas)[::-1]
    seed_idx = -1
    for idx in sorted_face_indices:
        if areas[idx] > 1e-8:
            seed_idx = idx
            break
            
    if seed_idx == -1:
        return []

    project_normal = normals[seed_idx].copy()
    tagged[seed_idx] = True
    
    while True:
        # Loop through all faces and group those within the half-angle limit
        dots = np.dot(normals, project_normal)
        group_mask = (dots > angle_limit_half_cos) & (areas > 1e-8)
        tagged[group_mask] = True
        
        group_indices = np.where(group_mask)[0]
        
        # Calculate the weighted average normal of this discovered group
        if len(group_indices) > 0:
            group_normals = normals[group_indices]
            group_areas = areas[group_indices]
            
            if area_weight <= 0.0:
                avg_norm = np.sum(group_normals, axis=0)
            elif area_weight >= 1.0:
                avg_norm = np.sum(group_normals * group_areas[:, np.newaxis], axis=0)
            else:
                area_blend = (group_areas * area_weight) + (1.0 - area_weight)
                avg_norm = np.sum(group_normals * area_blend[:, np.newaxis], axis=0)
                
            nlen = np.linalg.norm(avg_norm)
            if nlen > 1e-8:
                project_normal_array.append(avg_norm / nlen)
            else:
                project_normal_array.append(project_normal.copy())
        else:
            project_normal_array.append(project_normal.copy())
            
        # Find the most unique angle that points away from ALL existing discovered normals
        best_angle = 1.0
        best_angle_idx = -1
        
        unassigned_now = np.where(~tagged)[0]
        if len(unassigned_now) > 0:
            all_proj_normals = np.array(project_normal_array)
            dot_matrix = np.dot(normals[unassigned_now], all_proj_normals.T)
            max_dots = np.max(dot_matrix, axis=1)
            min_max_idx = np.argmin(max_dots)
            
            best_angle = max_dots[min_max_idx]
            best_angle_idx = unassigned_now[min_max_idx]
            
        if best_angle < angle_limit_cos and best_angle_idx != -1:
            # Continue search with this highly unique face as the next seed
            project_normal = normals[best_angle_idx].copy()
            tagged[best_angle_idx] = True
        else:
            # All remaining unassigned faces already fall within the projection limit 
            # of existing normals. Stop finding new axes.
            break

    return project_normal_array

    return project_normal_array

def get_island_diagonal_support(uvs, scale, rotation_radians):
    """
    Computes the 'support diagonal' of an island under rotation and scale.
    Matches Blender's PackIsland::get_diagonal_support logic.
    Returns (width_half, height_half) of the AABB in world space.
    """
    if rotation_radians == 0:
        scaled_uvs = uvs * scale
    else:
        c, s = np.cos(rotation_radians), np.sin(rotation_radians)
        # 2x2 rotation matrix
        rmat = np.array([[c, -s], [s, c]])
        scaled_uvs = uvs @ rmat.T * scale
    
    min_pt = np.min(scaled_uvs, axis=0)
    max_pt = np.max(scaled_uvs, axis=0)
    return (max_pt - min_pt) * 0.5


def pre_rotate_islands(island_uvs):
    """
    Rotates islands to minimize their AABB area.
    Heuristic: 'Stand up' islands so they are tall rather than wide.
    Matches Blender's PackIsland::calculate_pre_rotation_ logic.
    """
    rotated_islands = []
    for uvs in island_uvs:
        # Simplified minimum AABB: Try 90-degree increments or use hull
        # For Smart UV, islands are usually rectilinear.
        min_uv = np.min(uvs, axis=0)
        max_uv = np.max(uvs, axis=0)
        w, h = max_uv - min_uv
        
        # If width > height, rotate 90 deg to "stand up"
        if w > h:
            best_uvs = uvs.copy()
            # Rotate 90 CCW
            temp = best_uvs[:, 0].copy()
            best_uvs[:, 0] = -best_uvs[:, 1]
            best_uvs[:, 1] = temp
            best_uvs -= np.min(best_uvs, axis=0)
            rotated_islands.append(best_uvs)
        else:
            rotated_islands.append(uvs - min_uv)
    return rotated_islands

def pack_islands(island_uvs, margin=0.0, target_aspect=1.0):
    """
    Pure Python port of Blender's Alpaca (L-Packer) with 90-deg rotation and hole-filling.
    """
    if not island_uvs:
        return []

    # 1. Pre-process islands: normalize and record dimensions
    # Blender sorts by "biggest" (longest large edge)
    islands_raw = []
    for i, uvs in enumerate(island_uvs):
        m_min = np.min(uvs, axis=0)
        m_max = np.max(uvs, axis=0)
        w, h = m_max - m_min
        islands_raw.append({
            'idx': i,
            'uvs': uvs - m_min,
            'w': w,
            'h': h,
            'max_dim': max(w, h),
            'min_dim': min(w, h)
        })

    # Sort descending by max_dim (Blender-style)
    islands_raw.sort(key=lambda x: (x['max_dim'], x['min_dim']), reverse=True)

    # 2. Alpaca Packing State
    next_u1 = 0.0
    next_v1 = 0.0
    zigzag = next_u1 / target_aspect < next_v1 # Stripe direction
    
    # Track a single AABB "hole" which may be filled at any time.
    # Matches Blender's pack_islands_alpaca_rotate exactly.
    hole_u0, hole_v0 = 0.0, 0.0
    hole_dw, hole_dh = 0.0, 0.0 # du <= dv is enforced for hole_dw <= hole_dh
    hole_rotate = False

    def update_hole_rotate(u0, v0, u1, v1):
        nonlocal hole_u0, hole_v0, hole_dw, hole_dh, hole_rotate
        quad_area = (u1 - u0) * (v1 - v0)
        if quad_area <= (hole_dw * hole_dh):
            return
        hole_u0, hole_v0 = u0, v0
        hole_dw, hole_dh = u1 - u0, v1 - v0
        if hole_dh < hole_dw:
            hole_dw, hole_dh = hole_dh, hole_dw
            hole_rotate = True
        else:
            hole_rotate = False
    
    u0, v0 = 0.0, 0.0
    packed_results = [None] * len(island_uvs)
    
    print(f"[smart_uv] 6. Packing {len(island_uvs)} items with Alpaca...")
    
    for island in islands_raw:
        w_curr, h_curr = island['w'], island['h']
        min_dim, max_dim = island['min_dim'], island['max_dim']
        
        # 1. Try filling the single hole first
        if min_dim <= hole_dw and max_dim <= hole_dh:
            phi_rot = 0.0
            # Matches Blender: hole_rotate == (min_dim == island.uv_diagonal.x)
            if hole_rotate == (min_dim == w_curr):
                phi_rot = 90.0
                pw, ph = h_curr, w_curr
            else:
                phi_rot = 0.0
                pw, ph = w_curr, h_curr
            
            uvs = island['uvs'].copy()
            if phi_rot == 90.0:
                temp = uvs[:, 0].copy()
                uvs[:, 0] = -uvs[:, 1]
                uvs[:, 1] = temp
                uvs -= np.min(uvs, axis=0)
            
            uvs[:, 0] += hole_u0 + margin
            uvs[:, 1] += hole_v0 + margin
            packed_results[island['idx']] = uvs
            
            # Update hole leftovers (Blender style)
            p = [hole_u0, hole_v0, 
                 hole_u0 + (max_dim if hole_rotate else min_dim),
                 hole_v0 + (min_dim if hole_rotate else max_dim),
                 hole_u0 + (hole_dh if hole_rotate else hole_dw),
                 hole_v0 + (hole_dw if hole_rotate else hole_dh)]
            
            hole_dw, hole_dh = 0.0, 0.0 # invalidate
            update_hole_rotate(p[0], p[3], p[4], p[5])
            update_hole_rotate(p[2], p[1], p[4], p[5])
            continue
            
        # 2. Standard strip placement
        restart = False
        if zigzag:
            restart = (next_v1 < v0 + min_dim + 2*margin)
        else:
            restart = (next_u1 < u0 + min_dim + 2*margin)
            
        if restart:
            update_hole_rotate(u0, v0, next_u1, next_v1)
            zigzag = (next_u1 / target_aspect < next_v1)
            u0 = next_u1 if zigzag else 0.0
            v0 = 0.0 if zigzag else next_v1
            
        # Best 90-deg rotation for strip
        if zigzag == (min_dim == w_curr):
            phi_rot = 90.0
            pw, ph = h_curr + 2*margin, w_curr + 2*margin
        else:
            phi_rot = 0.0
            pw, ph = w_curr + 2*margin, h_curr + 2*margin
            
        uvs = island['uvs'].copy()
        if phi_rot == 90.0:
            temp = uvs[:, 0].copy()
            uvs[:, 0] = -uvs[:, 1]
            uvs[:, 1] = temp
            uvs -= np.min(uvs, axis=0)
            
        uvs[:, 0] += u0 + margin
        uvs[:, 1] += v0 + margin
        packed_results[island['idx']] = uvs
        
        if zigzag:
            v0 += ph
            next_u1 = max(next_u1, u0 + pw)
            next_v1 = max(next_v1, v0)
        else:
            u0 += pw
            next_v1 = max(next_v1, v0 + ph)
            next_u1 = max(next_u1, u0)

    # 3. Global normalization
    final_max = max(next_u1, next_v1)
    if final_max > 0:
        for i in range(len(packed_results)):
            packed_results[i] /= final_max
            
    return packed_results




def smart_uv_unwrap(vertices, faces, margin=0.01, angle_limit=1.15192, area_weight=0.0):
    import time
    """
    Python implementation of Blender's Smart UV Project.
    """
    num_faces = len(faces)
    num_vertices = len(vertices)
    
    angle_limit_cos = np.cos(angle_limit)
    angle_limit_half_cos = np.cos(angle_limit / 2.0)
    
    # 1. Calculate Normals and Areas
    print("[smart_uv] 1. Calculating Normals and Areas...")
    t_start = time.time()
    normals, areas = get_face_normals_and_areas(vertices, faces)
    print(f"[smart_uv] 1. Finished in {time.time() - t_start:.2f}s")
    
    # Ignore zero area faces
    valid_face_mask = areas > 1e-8
    
    # 2. Calculate Projection Normals
    print("[smart_uv] 2. Calculating Projection Normals...")
    t_start = time.time()
    project_normals = calculate_project_normals(
        normals[valid_face_mask], 
        areas[valid_face_mask], 
        angle_limit_half_cos, 
        angle_limit_cos, 
        area_weight
    )
    print(f"[smart_uv] 2. Finished in {time.time() - t_start:.2f}s, found {len(project_normals)} project normals")
    
    if not project_normals:
        # Fallback: simple projection
        project_normals = [np.array([0.0, 0.0, 1.0])]
        
    # 3. Assign faces to the best projection normal with Stability Smoothing
    print("[smart_uv] 3. Assigning faces (with Stability Smoothing)...")
    t_start = time.time()
    
    # Original dot product matrix [N_faces, num_proj_normals]
    proj_normals_array = np.array(project_normals)
    # Filter valid faces only for the score calculation to save time
    valid_face_indices = np.where(valid_face_mask)[0]
    dot_matrix_full = np.zeros((num_faces, len(project_normals)), dtype=np.float32)
    dot_matrix_full[valid_face_mask] = np.dot(normals[valid_face_mask], proj_normals_array.T)
    
    # 4. Build Face Adjacency Graph (used for both smoothing and island building)
    print("[smart_uv] 4. Building Face Adjacency Graph...")
    # Quantize for robustness against unwelded meshes
    coords_quantized = (vertices * 5e4).astype(np.int64)
    _, unique_v_indices = np.unique(coords_quantized, axis=0, return_inverse=True)
    weld_faces = unique_v_indices[faces]
    
    v0, v1, v2 = weld_faces[:, 0], weld_faces[:, 1], weld_faces[:, 2]
    e1, e2, e3 = np.column_stack([v0, v1]), np.column_stack([v1, v2]), np.column_stack([v2, v0])
    all_edges = np.vstack([e1, e2, e3])
    all_edges.sort(axis=1)
    edge_hash = (all_edges[:, 0].astype(np.uint64) << 32) + all_edges[:, 1].astype(np.uint64)
    sort_idx = np.argsort(edge_hash)
    edge_hash_sorted = edge_hash[sort_idx]
    face_idx_sorted = np.tile(np.arange(num_faces), 3)[sort_idx]
    match_mask = (edge_hash_sorted[:-1] == edge_hash_sorted[1:])
    adj_f1, adj_f2 = face_idx_sorted[:-1][match_mask], face_idx_sorted[1:][match_mask]

    # Create adjacency matrix for smoothing
    adj_weights = np.ones(len(adj_f1), dtype=np.float32)
    W = csr_matrix((adj_weights, (adj_f1, adj_f2)), shape=(num_faces, num_faces))
    W = W + W.T # Symmetric
    
    # Normalized Laplacian smoothing pass
    # W_norm = D^-1 * W, where D is diagonal degree matrix
    row_sums = np.array(W.sum(axis=1)).flatten()
    inv_D = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    W_norm = W.multiply(inv_D[:, np.newaxis])
    
    # 80-pass smoothing:consensus between neighbors
    smoothed_dots = dot_matrix_full
    for _ in range(80):
         smoothed_dots = 0.1 * smoothed_dots + 0.9 * W_norm.dot(smoothed_dots) # Very heavy neighbor weight
         
    best_proj_idx = np.argmax(smoothed_dots, axis=1)
    
    # Final projection index per face
    proj_idx_per_face = np.where(valid_face_mask, best_proj_idx, -1)
    
    # --- STEP 4c: Iterative Small Island Merging ---
    # Running multiple passes ensures that tiny islands merged into each other 
    # are re-evaluated and merged into larger ones.
    for merge_pass in range(5):
        # Find CC
        valid_adj = (proj_idx_per_face[adj_f1] == proj_idx_per_face[adj_f2]) & (proj_idx_per_face[adj_f1] != -1)
        data_i = np.ones(np.sum(valid_adj), dtype=bool)
        adj_f1_i, adj_f2_i = adj_f1[valid_adj], adj_f2[valid_adj]
        adj_matrix_i = csr_matrix((data_i, (adj_f1_i, adj_f2_i)), shape=(num_faces, num_faces))
        n_curr, labels_curr = connected_components(csgraph=adj_matrix_i, directed=False)
        
        # Identify small islands (< 500 faces)
        label_counts = np.bincount(labels_curr)
        tiny_labels = np.where((label_counts < 500) & (label_counts > 0))[0]
        
        if len(tiny_labels) == 0:
            break
            
        print(f"[smart_uv] Merge Pass {merge_pass+1}: Vectorised merging for {len(tiny_labels)} tiny islands...")
        is_tiny_label = np.zeros(n_curr, dtype=bool)
        is_tiny_label[tiny_labels] = True
        face_is_tiny = is_tiny_label[labels_curr]
        
        tiny_edge_mask = face_is_tiny[adj_f1] | face_is_tiny[adj_f2]
        te_f1, te_f2 = adj_f1[tiny_edge_mask], adj_f2[tiny_edge_mask]
        
        t_f = np.concatenate([te_f1[face_is_tiny[te_f1]], te_f2[face_is_tiny[te_f2]]])
        n_f = np.concatenate([te_f2[face_is_tiny[te_f1]], te_f1[face_is_tiny[te_f2]]])
        
        t_island_ids = labels_curr[t_f]
        n_axes = proj_idx_per_face[n_f]
        
        v_n_mask = n_axes != -1
        if not np.any(v_n_mask): break
        
        t_ids_v = t_island_ids[v_n_mask]
        n_axes_v = n_axes[v_n_mask]
        
        combined = (t_ids_v.astype(np.uint64) << 32) | n_axes_v.astype(np.uint64)
        keys, counts = np.unique(combined, return_counts=True)
        uids, uaxes = (keys >> 32).astype(np.int32), (keys & 0xFFFFFFFF).astype(np.int32)
        
        s_idx = np.lexsort((counts, uids))
        uids_s, uaxes_s = uids[s_idx], uaxes[s_idx]
        _, first_idx = np.unique(uids_s, return_index=True)
        last_idx = np.concatenate([first_idx[1:] - 1, [len(uids_s) - 1]])
        
        best_axis_map = np.full(n_curr, -1, dtype=np.int32)
        best_axis_map[uids_s[last_idx]] = uaxes_s[last_idx]
        
        can_update = face_is_tiny & (best_axis_map[labels_curr] != -1)
        proj_idx_per_face[can_update] = best_axis_map[labels_curr[can_update]]
    

    # 4b. Final Connected Components (Islands)
    # Only connect faces that share an edge AND share the same projection axis
    valid_adj_final = (proj_idx_per_face[adj_f1] == proj_idx_per_face[adj_f2]) & (proj_idx_per_face[adj_f1] != -1)
    adj_f1_final, adj_f2_final = adj_f1[valid_adj_final], adj_f2[valid_adj_final]
    
    data_final = np.ones(len(adj_f1_final), dtype=bool)
    adj_matrix_final = csr_matrix((data_final, (adj_f1_final, adj_f2_final)), shape=(num_faces, num_faces))
    
    t_cc = time.time()
    n_components, labels = connected_components(csgraph=adj_matrix_final, directed=False)
    
    # Reassemble islands
    islands = []
    sort_by_label = np.argsort(labels)
    sorted_labels = labels[sort_by_label]
    _, label_starts = np.unique(sorted_labels, return_index=True)
    
    for i in range(len(label_starts)):
        start = label_starts[i]
        end = label_starts[i+1] if i + 1 < len(label_starts) else num_faces
        island_faces = sort_by_label[start:end]
        
        f_idx = island_faces[0]
        p_idx = proj_idx_per_face[f_idx]
        
        if p_idx != -1: # Only valid faces
            islands.append({
                'proj_idx': p_idx,
                'faces': island_faces
            })
            
    print(f"[smart_uv] 3 & 4. Consolidated Grouping finished in {time.time() - t_start:.2f}s, found {len(islands)} islands")
        
    # Project each island
    print("[smart_uv] 5. Projecting islands...")
    t_start = time.time()
    island_uv_arrays = []
    
    # Pre-compute matrices for all unique projection normals used
    # This prevents running the dominant axis math in a loop for each island
    unique_proj_indices = sorted(list(set([island['proj_idx'] for island in islands])))
    proj_matrices = {}
    for p_idx in unique_proj_indices:
        p_norm = project_normals[p_idx]
        proj_matrices[p_idx] = axis_dominant_v3_to_m3(p_norm)
        
    for island in islands:
        # Create the projection matrix
        mat = proj_matrices[island['proj_idx']]
        
        island_faces_indices = island['faces']
        # (num_island_faces, 3, 3) 3D vertices
        island_verts = vertices[faces[island_faces_indices]]
        
        # Map 3D to 2D using the projection matrix
        # Einsum for Nx3x3 vertices * 3x3 matrix -> Nx3x3 projection
        projected = np.einsum('ij,nmj->nmi', mat, island_verts) # (N, 3, 3)
        island_uvs_2d = projected[:, :, :2] # Take X, Y. Drop Z.
        
        island_uv_arrays.append(island_uvs_2d.reshape(-1, 2))
    print(f"[smart_uv] 5. Finished projection in {time.time() - t_start:.2f}s")
    
    # 5b. Pre-rotate islands
    print("[smart_uv] 5b. Pre-rotating islands for minimum AABB...")
    t_start_pr = time.time()
    island_uv_arrays = pre_rotate_islands(island_uv_arrays)
    print(f"[smart_uv] 5b. Finished pre-rotation in {time.time() - t_start_pr:.2f}s")
        
    # 6. Pack islands
    print("[smart_uv] 6. Packing islands with Alpaca...")
    t_start_pack = time.time()
    packed_uvs_flat = pack_islands(island_uv_arrays, margin=margin)
    print(f"[smart_uv] 6. Finished packing in {time.time() - t_start_pack:.2f}s")
    
    # Reassemble into flat format
    print("[smart_uv] 7. Reassembling and Deduping Flat UVs...")
    t_start = time.time()
    flat_uvs = np.zeros((num_faces * 3, 2), dtype=np.float32)
    for island, packed in zip(islands, packed_uvs_flat):
         face_indices = island['faces']
         packed = packed.reshape(-1, 3, 2)
         for idx, f_idx in enumerate(face_indices):
             flat_uvs[f_idx*3:(f_idx+1)*3] = packed[idx]
             
    # Deduplicate exact matching vertices + UVs to recreate Trimesh split format
    flat_faces = faces.reshape(-1)
    quantized_uvs = (flat_uvs * 1e5).astype(np.int32)
    
    pack = np.column_stack((flat_faces, quantized_uvs))
    
    unique_pairs, unique_indices, inverse_indices = np.unique(pack, axis=0, return_index=True, return_inverse=True)
    
    vmap = flat_faces[unique_indices].astype(np.int32)
    new_vertices = vertices[vmap]
    new_uvs = flat_uvs[unique_indices]
    
    # Clamp UVs strictly to [0, 1] to prevent nvdiffrast clipping outside viewport
    new_uvs = np.clip(new_uvs, 0.0, 1.0)
    
    new_faces = inverse_indices.reshape(num_faces, 3).astype(np.int32)
    
    print(f"[smart_uv] 7. Finished Deduping in {time.time() - t_start:.2f}s")
    
    return new_vertices, new_faces, new_uvs, vmap
