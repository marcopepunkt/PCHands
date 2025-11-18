from klampt import vis, Geometry3D

# Load STL
geom = Geometry3D()
geom.loadFile("/home/marco/PCHands/assets/orca_v1/mesh/converted_CarpalsAssembly.stl")

# Visualize
vis.add("model", geom)
vis.show()
vis.spin(float('inf'))  # keep window open
