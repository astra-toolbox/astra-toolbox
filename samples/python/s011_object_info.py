import astra

# Create two volume geometries
vol_geom1 = astra.create_vol_geom(256, 256)
vol_geom2 = astra.create_vol_geom(512, 256)

# Create volumes
v0 = astra.data2d.create('-vol', vol_geom1)
v1 = astra.data2d.create('-vol', vol_geom2)
v2 = astra.data2d.create('-vol', vol_geom2)

# Show the currently allocated volumes
astra.data2d.info()


astra.data2d.delete(v2)
astra.data2d.info()

astra.data2d.clear()
astra.data2d.info()



# The same clear and info command also work for other object types:
astra.algorithm.info()  
astra.data3d.info()
astra.projector.info()
astra.matrix.info()
