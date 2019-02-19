import numpy as np
import unittest
import astra
import math
import pylab

# return length of intersection of the line through points src = (x,y)
# and det (x,y), and the rectangle defined by xmin, ymin, xmax, ymax
#
# TODO: Generalize from 2D to n-dimensional
def intersect_line_rectangle(src, det, xmin, xmax, ymin, ymax):
  EPS = 1e-5

  if np.abs(src[0] - det[0]) < EPS:
    if src[0] >= xmin and src[0] < xmax:
      return ymax - ymin
    else:
      return 0.0
  if np.abs(src[1] - det[1]) < EPS:
    if src[1] >= ymin and src[1] < ymax:
      return xmax - xmin
    else:
      return 0.0

  n = np.sqrt((det[0] - src[0]) ** 2 + (det[1] - src[1]) ** 2)

  check = [ (-(xmin - src[0]), -(det[0] - src[0]) / n ),
            (xmax - src[0], (det[0] - src[0]) / n ),
            (-(ymin - src[1]), -(det[1] - src[1]) / n ),
            (ymax - src[1], (det[1] - src[1]) / n ) ]

  pre = [ -np.Inf ]
  post = [ np.Inf ]

  for p, q in check:
    r = p / (1.0 * q)
    if q > 0:
      post.append(r)    # exiting half-plane
    else:
      pre.append(r)     # entering half-plane

  end_r = np.min(post)
  start_r = np.max(pre)

  if end_r > start_r:
    return end_r - start_r
  else:
    return 0.0

def intersect_line_rectangle_feather(src, det, xmin, xmax, ymin, ymax, feather):
  return intersect_line_rectangle(src, det,
                                  xmin-feather, xmax+feather,
                                  ymin-feather, ymax+feather)

def intersect_line_rectangle_interval(src, det, xmin, xmax, ymin, ymax, f):
  a = intersect_line_rectangle_feather(src, det, xmin, xmax, ymin, ymax, -f)
  b = intersect_line_rectangle(src, det, xmin, xmax, ymin, ymax)
  c = intersect_line_rectangle_feather(src, det, xmin, xmax, ymin, ymax, f)
  return (a,b,c)


# x-coord of intersection of the line (src, det) with the horizontal line at y
def intersect_line_horizontal(src, det, y):
  EPS = 1e-5

  if np.abs(src[1] - det[1]) < EPS:
    return np.nan

  t = (y - src[1]) / (det[1] - src[1])

  return src[0] + t * (det[0] - src[0])


# length of the intersection of the strip with boundaries edge1, edge2 with the horizontal
# segment at y, with horizontal extent x_seg
def intersect_ray_horizontal_segment(edge1, edge2, y, x_seg):
  e1 = intersect_line_horizontal(edge1[0], edge1[1], y)
  e2 = intersect_line_horizontal(edge2[0], edge2[1], y)

  if not (np.isfinite(e1) and np.isfinite(e2)):
    return np.nan

  (e1, e2) = np.sort([e1, e2])
  (x1, x2) = np.sort(x_seg)
  l = np.max([e1, x1])
  r = np.min([e2, x2])
  #print(edge1, edge2, y, x_seg, r-l)
  return np.max([r-l, 0.0])

def intersect_ray_vertical_segment(edge1, edge2, x, y_seg):
  # mirror edge1 and edge2
  edge1 = [ (a[1], a[0]) for a in edge1 ]
  edge2 = [ (a[1], a[0]) for a in edge2 ]
  return intersect_ray_horizontal_segment(edge1, edge2, x, y_seg)






# LINE GENERATORS
# ---------------
#
# Per ray these yield three lines, at respectively the center and two edges of the detector pixel.
# Each line is given by two points on the line.
# ( ( (p0x, p0y), (q0x, q0y) ), ( (p1x, p1y), (q1x, q1y) ), ( (p2x, p2y), (q2x, q2y) ) )

def gen_lines_fanflat(proj_geom):
  angles = proj_geom['ProjectionAngles']
  for theta in angles:
    #theta = -theta
    src = ( math.sin(theta) * proj_geom['DistanceOriginSource'],
           -math.cos(theta) * proj_geom['DistanceOriginSource'] )
    detc= (-math.sin(theta) * proj_geom['DistanceOriginDetector'],
            math.cos(theta) * proj_geom['DistanceOriginDetector'] )
    detu= ( math.cos(theta) * proj_geom['DetectorWidth'],
            math.sin(theta) * proj_geom['DetectorWidth'] )

    src = np.array(src, dtype=np.float64)
    detc= np.array(detc, dtype=np.float64)
    detu= np.array(detu, dtype=np.float64)

    detb= detc + (0.5 - 0.5*proj_geom['DetectorCount']) * detu

    for i in range(proj_geom['DetectorCount']):
      yield ((src, detb + i * detu),
             (src, detb + (i - 0.5) * detu),
             (src, detb + (i + 0.5) * detu))

def gen_lines_fanflat_vec(proj_geom):
  v = proj_geom['Vectors']
  for i in range(v.shape[0]):
    src = v[i,0:2]
    detc = v[i,2:4]
    detu = v[i,4:6]

    detb = detc + (0.5 - 0.5*proj_geom['DetectorCount']) * detu
    for i in range(proj_geom['DetectorCount']):
      yield ((src, detb + i * detu),
             (src, detb + (i - 0.5) * detu),
             (src, detb + (i + 0.5) * detu))

def gen_lines_parallel(proj_geom):
  angles = proj_geom['ProjectionAngles']
  for theta in angles:
    ray = ( math.sin(theta),
           -math.cos(theta) )
    detc= (0, 0 )
    detu= ( math.cos(theta) * proj_geom['DetectorWidth'],
            math.sin(theta) * proj_geom['DetectorWidth'] )

    ray = np.array(ray, dtype=np.float64)
    detc= np.array(detc, dtype=np.float64)
    detu= np.array(detu, dtype=np.float64)


    detb= detc + (0.5 - 0.5*proj_geom['DetectorCount']) * detu

    for i in range(proj_geom['DetectorCount']):
      yield ((detb + i * detu - ray, detb + i * detu),
             (detb + (i - 0.5) * detu - ray, detb + (i - 0.5) * detu),
             (detb + (i + 0.5) * detu - ray, detb + (i + 0.5) * detu))


def gen_lines_parallel_vec(proj_geom):
  v = proj_geom['Vectors']
  for i in range(v.shape[0]):
    ray = v[i,0:2]
    detc = v[i,2:4]
    detu = v[i,4:6]

    detb = detc + (0.5 - 0.5*proj_geom['DetectorCount']) * detu

    for i in range(proj_geom['DetectorCount']):
      yield ((detb + i * detu - ray, detb + i * detu),
             (detb + (i - 0.5) * detu - ray, detb + (i - 0.5) * detu),
             (detb + (i + 0.5) * detu - ray, detb + (i + 0.5) * detu))


def gen_lines(proj_geom):
  g = { 'fanflat': gen_lines_fanflat,
        'fanflat_vec': gen_lines_fanflat_vec,
        'parallel': gen_lines_parallel,
        'parallel_vec': gen_lines_parallel_vec }
  for l in g[proj_geom['type']](proj_geom):
    yield l

range2d = ( 8, 64 )


def gen_random_geometry_fanflat():
  pg = astra.create_proj_geom('fanflat', 0.6 + 0.8 * np.random.random(), np.random.randint(*range2d), np.linspace(0, 2*np.pi, np.random.randint(*range2d), endpoint=False), 256 * (0.5 + np.random.random()), 256 * np.random.random())
  return pg

def gen_random_geometry_parallel():
  pg = astra.create_proj_geom('parallel', 0.8 + 0.4 * np.random.random(), np.random.randint(*range2d), np.linspace(0, 2*np.pi, np.random.randint(*range2d), endpoint=False))
  return pg

def gen_random_geometry_fanflat_vec():
  Vectors = np.zeros([16,6])
  # We assume constant detector width in these tests
  w = 0.6 + 0.8 * np.random.random()
  for i in range(Vectors.shape[0]):
    angle1 = 2*np.pi*np.random.random()
    angle2 = angle1 + 0.5 * np.random.random()
    dist1 = 256 * (0.5 + np.random.random())
    detc = 10 * np.random.random(size=2)
    detu = [ math.cos(angle1) * w, math.sin(angle1) * w ]
    src = [ math.sin(angle2) * dist1, -math.cos(angle2) * dist1 ]
    Vectors[i, :] = [ src[0], src[1], detc[0], detc[1], detu[0], detu[1] ]
  pg = astra.create_proj_geom('fanflat_vec', np.random.randint(*range2d), Vectors)

  # TODO: Randomize more
  pg = astra.create_proj_geom('fanflat_vec', np.random.randint(*range2d), Vectors)
  return pg

def gen_random_geometry_parallel_vec():
  Vectors = np.zeros([16,6])
  # We assume constant detector width in these tests
  w = 0.6 + 0.8 * np.random.random()
  for i in range(Vectors.shape[0]):
    l = 0.6 + 0.8 * np.random.random()
    angle1 = 2*np.pi*np.random.random()
    angle2 = angle1 + 0.5 * np.random.random()
    detc = 10 * np.random.random(size=2)
    detu = [ math.cos(angle1) * w, math.sin(angle1) * w ]
    ray = [ math.sin(angle2) * l, -math.cos(angle2) * l ]
    Vectors[i, :] = [ ray[0], ray[1], detc[0], detc[1], detu[0], detu[1] ]
  pg = astra.create_proj_geom('parallel_vec', np.random.randint(*range2d), Vectors)
  return pg




nloops = 50
seed = 123

class Test2DKernel(unittest.TestCase):
  def single_test(self, type, proj_type):
      shape = np.random.randint(*range2d, size=2)
      # these rectangles are biased, but that shouldn't matter
      rect_min = [ np.random.randint(0, a) for a in shape ]
      rect_max = [ np.random.randint(rect_min[i]+1, shape[i]+1) for i in range(len(shape))]
      if True:
          #pixsize = 0.5 + np.random.random(size=2)
          pixsize = np.array([0.5, 0.5]) + np.random.random()
          origin = 10 * np.random.random(size=2)
      else:
          pixsize = (1.,1.)
          origin = (0.,0.)
      vg = astra.create_vol_geom(shape[1], shape[0],
                                 origin[0] - 0.5 * shape[0] * pixsize[0],
                                 origin[0] + 0.5 * shape[0] * pixsize[0],
                                 origin[1] - 0.5 * shape[1] * pixsize[1],
                                 origin[1] + 0.5 * shape[1] * pixsize[1])
      #print(vg)

      if type == 'parallel':
        pg = gen_random_geometry_parallel()
        projector_id = astra.create_projector(proj_type, pg, vg)
      elif type == 'parallel_vec':
        pg = gen_random_geometry_parallel_vec()
        projector_id = astra.create_projector(proj_type, pg, vg)
      elif type == 'fanflat':
        pg = gen_random_geometry_fanflat()
        projector_id = astra.create_projector(proj_type + '_fanflat', pg, vg)
      elif type == 'fanflat_vec':
        pg = gen_random_geometry_fanflat_vec()
        projector_id = astra.create_projector(proj_type + '_fanflat', pg, vg)


      data = np.zeros((shape[1], shape[0]), dtype=np.float32)
      data[rect_min[1]:rect_max[1],rect_min[0]:rect_max[0]] = 1

      sinogram_id, sinogram = astra.create_sino(data, projector_id)

      #print(pg)
      #print(vg)

      astra.data2d.delete(sinogram_id)

      astra.projector.delete(projector_id)

      if proj_type == 'line':

        a = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        b = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        c = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)

        i = 0
        #print( origin[0] + (-0.5 * shape[0] + rect_min[0]) * pixsize[0], origin[0] + (-0.5 * shape[0] + rect_max[0]) * pixsize[0], origin[1] + (-0.5 * shape[1] + rect_min[1]) * pixsize[1], origin[1] + (-0.5 * shape[1] + rect_max[1]) * pixsize[1])
        for center, edge1, edge2 in gen_lines(pg):
          (src, det) = center
          #print(src,det)

          # NB: Flipped y-axis here, since that is how astra interprets 2D volumes
          # We compute line intersections with slightly bigger (cw) and
          # smaller (aw) rectangles, and see if the kernel falls
          # between these two values.
          (aw,bw,cw) = intersect_line_rectangle_interval(src, det,
                        origin[0] + (-0.5 * shape[0] + rect_min[0]) * pixsize[0],
                        origin[0] + (-0.5 * shape[0] + rect_max[0]) * pixsize[0],
                        origin[1] + (+0.5 * shape[1] - rect_max[1]) * pixsize[1],
                        origin[1] + (+0.5 * shape[1] - rect_min[1]) * pixsize[1],
                        1e-3)
          a[i] = aw
          b[i] = bw
          c[i] = cw
          i += 1
        # Add weight for pixel / voxel size
        try:
          detweight = pg['DetectorWidth']
        except KeyError:
          detweight = np.sqrt(pg['Vectors'][0,4]*pg['Vectors'][0,4] + pg['Vectors'][0,5]*pg['Vectors'][0,5] )
        a *= detweight
        b *= detweight
        c *= detweight
        a = a.reshape(astra.functions.geom_size(pg))
        b = b.reshape(astra.functions.geom_size(pg))
        c = c.reshape(astra.functions.geom_size(pg))

        # Check if sinogram lies between a and c
        y = np.min(sinogram-a)
        z = np.min(c-sinogram)
        x = np.max(np.abs(sinogram-b)) # ideally this is small, but can be large
                                       # due to discontinuities in line kernel
        self.assertFalse(z < 0 or y < 0)
        if z < 0 or y < 0:
          print(y,z,x)
          pylab.gray()
          pylab.imshow(data)
          pylab.figure()
          pylab.imshow(sinogram)
          pylab.figure()
          pylab.imshow(b)
          pylab.figure()
          pylab.imshow(a)
          pylab.figure()
          pylab.imshow(c)
          pylab.figure()
          pylab.imshow(sinogram-a)
          pylab.figure()
          pylab.imshow(c-sinogram)
          pylab.show()
      elif proj_type == 'distance_driven':
        a = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        i = 0
        for center, edge1, edge2 in gen_lines(pg):
          (xd, yd) = center[1] - center[0]
          l = 0.0
          if np.abs(xd) > np.abs(yd): # horizontal ray
            y_seg = (origin[1] + (+0.5 * shape[1] - rect_max[1]) * pixsize[1],
                     origin[1] + (+0.5 * shape[1] - rect_min[1]) * pixsize[1])
            for j in range(rect_min[0], rect_max[0]):
              x = origin[0] + (-0.5 * shape[0] + j + 0.5) * pixsize[0]
              l += intersect_ray_vertical_segment(edge1, edge2, x, y_seg) * pixsize[0]
          else:
            x_seg = (origin[0] + (-0.5 * shape[0] + rect_max[0]) * pixsize[0],
                     origin[0] + (-0.5 * shape[0] + rect_min[0]) * pixsize[0])
            for j in range(rect_min[1], rect_max[1]):
              y = origin[1] + (+0.5 * shape[1] - j - 0.5) * pixsize[1]
              l += intersect_ray_horizontal_segment(edge1, edge2, y, x_seg) * pixsize[1]
          a[i] = l
          i += 1
        a = a.reshape(astra.functions.geom_size(pg))
        x = np.max(np.abs(sinogram-a))
        if x > 2e-3:
          pylab.gray()
          pylab.imshow(data)
          pylab.figure()
          pylab.imshow(sinogram)
          pylab.figure()
          pylab.imshow(a)
          pylab.figure()
          pylab.imshow(sinogram-a)
          pylab.show()
        self.assertFalse(x > 2e-3)


  def test_par(self):
    np.random.seed(seed)
    for _ in range(nloops):
      self.single_test('parallel', 'line')
  def test_par_dd(self):
    np.random.seed(seed)
    for _ in range(nloops):
      self.single_test('parallel', 'distance_driven')
  def test_fan(self):
    np.random.seed(seed)
    for _ in range(nloops):
      self.single_test('fanflat', 'line')
  def test_parvec(self):
    np.random.seed(seed)
    for _ in range(nloops):
      self.single_test('parallel_vec', 'line')
  def test_parvec_dd(self):
    np.random.seed(seed)
    for _ in range(nloops):
      self.single_test('parallel_vec', 'distance_driven')
  def test_fanvec(self):
    np.random.seed(seed)
    for _ in range(nloops):
      self.single_test('fanflat_vec', 'line')




if __name__ == '__main__':
  unittest.main()

#print(intersect_line_rectangle((0.,-256.),(-27.,0.),11.6368454385 20.173128227 3.18989047649 5.62882841606)
