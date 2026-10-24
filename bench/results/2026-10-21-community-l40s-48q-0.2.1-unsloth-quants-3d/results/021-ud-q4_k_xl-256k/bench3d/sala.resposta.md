```python
import bpy


def add_box(name, dimensions, location):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.dimensions = dimensions
    return obj


# Piso: 4 m (X) × 5 m (Y) × 0,1 m de espessura
# Face superior em z = 0, portanto o centro fica em z = -0,05
add_box("Piso", (4.0, 5.0, 0.1), (0.0, 0.0, -0.05))

# Paredes: 0,1 m de espessura e 2,5 m de altura, de z = 0 a z = 2,5
# Centro vertical em z = 1,25

# Paredes paralelas ao eixo X, com faces externas em y = ±2,5
add_box("Parede_Y_Positivo", (4.0, 0.1, 2.5), (0.0, 2.45, 1.25))
add_box("Parede_Y_Negativo", (4.0, 0.1, 2.5), (0.0, -2.45, 1.25))

# Paredes paralelas ao eixo Y, com faces externas em x = ±2
# Comprimento ajustado para encaixar entre as paredes paralelas ao eixo X
add_box("Parede_X_Positivo", (0.1, 4.8, 2.5), (1.95, 0.0, 1.25))
add_box("Parede_X_Negativo", (0.1, 4.8, 2.5), (-1.95, 0.0, 1.25))
```