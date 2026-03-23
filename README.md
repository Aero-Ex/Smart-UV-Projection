# Smart-UV-Projection

A high-performance, pure Python implementation of Blender's **Smart UV Project** algorithm and the **Alpaca** (Skyline) bin-packing engine.

## Features
- **Pure Python**: No dependency on `bpy` or Blender.
- **Blender-Equivalent Logic**: Perfectly ports the projection normal calculation and face grouping.
- **Consolidation**: Laplacian smoothing and iterative merging to minimize island counts.
- **Alpaca Packing**: Port of Blender's native C++ packing engine (`uv_pack.cc`) for dense, efficient layouts.
- **Fast**: Processed 2M+ face meshes in ~15 seconds.

## Installation
```bash
pip install Smart-UV-Projection
```

## Usage
```python
import trimesh
from smart_uv import smart_uv_unwrap

# Load your mesh
mesh = trimesh.load("model.obj")

# Unwrap
new_vertices, new_faces, new_uvs, vmap = smart_uv_unwrap(
    mesh.vertices, 
    mesh.faces, 
    angle_limit=1.1519, # ~66 degrees
    margin=0.01
)
```

## License
MIT
