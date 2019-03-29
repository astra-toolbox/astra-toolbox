import numpy as np
import unittest
import astra
import math
import pylab

# Display sinograms with mismatch on test failure
DISPLAY=False

NONUNITDET=False
OBLIQUE=False
FLEXVOL=False
NONSQUARE=False  # non-square pixels not supported yet by most projectors

# Round interpolation weight to 8 bits to emulate CUDA texture unit precision
CUDA_8BIT_LINEAR=True
CUDA_TOL=2e-2

nloops = 50
seed = 123


# FAILURES:
# fan/cuda with flexible volume
# detweight for fan/cuda
# fan/strip relatively high numerical errors?
# parvec/line+linear for oblique

# INCONSISTENCY:
# effective_detweight vs norm(detu) in line/linear (oblique)



# return length of intersection of the line through points src = (x,y)
# and det (x,y), and the rectangle defined by xmin, ymin, xmax, ymax
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

# y-coord of intersection of the line (src, det) with the vertical line at x
def intersect_line_vertical(src, det, x):
  src = ( src[1], src[0] )
  det = ( det[1], det[0] )
  return intersect_line_horizontal(src, det, x)

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
  return np.max([r-l, 0.0])

def intersect_ray_vertical_segment(edge1, edge2, x, y_seg):
  # mirror edge1 and edge2
  edge1 = [ (a[1], a[0]) for a in edge1 ]
  edge2 = [ (a[1], a[0]) for a in edge2 ]
  return intersect_ray_horizontal_segment(edge1, edge2, x, y_seg)

# weight of the intersection of line with the horizontal segment at y, with horizontal extent x_seg
# using linear interpolation
def intersect_line_horizontal_segment_linear(src, det, y, x_seg, inter_width):
  EPS = 1e-5
  x = intersect_line_horizontal(src, det, y)

  assert(x_seg[1] - x_seg[0] + EPS >= inter_width)
  if x < x_seg[0] - 0.5*inter_width:
    return 0.0
  elif x < x_seg[0] + 0.5*inter_width:
    return (x - (x_seg[0] - 0.5*inter_width)) / inter_width
  elif x < x_seg[1] - 0.5*inter_width:
    return 1.0
  elif x < x_seg[1] + 0.5*inter_width:
    return (x_seg[1] + 0.5*inter_width - x) / inter_width
  else:
    return 0.0

def intersect_line_vertical_segment_linear(src, det, x, y_seg, inter_height):
  src = ( src[1], src[0] )
  det = ( det[1], det[0] )
  return intersect_line_horizontal_segment_linear(src, det, x, y_seg, inter_height)



def area_signed(a, b):
  return a[0] * b[1] - a[1] * b[0]

# is c to the left of ab
def is_left_of(a, b, c):
  EPS = 1e-5
  return area_signed( (b[0] - a[0], b[1] - a[1]), (c[0] - a[0], c[1] - a[1]) ) > EPS

# compute area of rect on left side of line
def halfarea_rect_line(src, det, xmin, xmax, ymin, ymax):
  pts = ( (xmin,ymin), (xmin,ymax), (xmax,ymin), (xmax,ymax) )
  pts_left = list(filter( lambda p: is_left_of(src, det, p), pts ))
  npts_left = len(pts_left)
  if npts_left == 0:
    return 0.0
  elif npts_left == 1:
    # triangle
    p = pts_left[0]
    xd = intersect_line_horizontal(src, det, p[1]) - p[0]
    yd = intersect_line_vertical(src, det, p[0]) - p[1]
    ret = 0.5 * abs(xd) * abs(yd)
    return ret
  elif npts_left == 2:
    p = pts_left[0]
    q = pts_left[1]
    if p[0] == q[0]:
      # vertical intersection
      x1 = intersect_line_horizontal(src, det, p[1]) - p[0]
      x2 = intersect_line_horizontal(src, det, q[1]) - q[0]
      ret = 0.5 * (ymax - ymin) * (abs(x1) + abs(x2))
      return ret
    else:
      assert(p[1] == q[1])
      # horizontal intersection
      y1 = intersect_line_vertical(src, det, p[0]) - p[1]
      y2 = intersect_line_vertical(src, det, q[0]) - q[1]
      ret = 0.5 * (xmax - xmin) * (abs(y1) + abs(y2))
      return ret
  else:
    # mirror and invert
    ret = ((xmax - xmin) * (ymax - ymin)) - halfarea_rect_line(det, src, xmin, xmax, ymin, ymax)
    return ret

# area of intersection of the strip with boundaries edge1, edge2 with rectangle
def intersect_ray_rect(edge1, edge2, xmin, xmax, ymin, ymax):
  s1 = halfarea_rect_line(edge1[0], edge1[1], xmin, xmax, ymin, ymax)
  s2 = halfarea_rect_line(edge2[0], edge2[1], xmin, xmax, ymin, ymax)
  return abs(s1 - s2)


# width of projection of detector orthogonal to ray direction
# i.e., effective detector width
def effective_detweight(src, det, u):
  ray = np.array(det) - np.array(src)
  ray = ray / np.linalg.norm(ray, ord=2)
  return abs(area_signed(ray, u))


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
  if not NONUNITDET:
    w = 1.0
  else:
    w = 0.6 + 0.8 * np.random.random()
  pg = astra.create_proj_geom('fanflat', w, np.random.randint(*range2d), np.linspace(0, 2*np.pi, np.random.randint(*range2d), endpoint=False), 256 * (0.5 + np.random.random()), 256 * np.random.random())
  return pg

def gen_random_geometry_parallel():
  if not NONUNITDET:
    w = 1.0
  else:
    w = 0.8 + 0.4 * np.random.random()
  pg = astra.create_proj_geom('parallel', w, np.random.randint(*range2d), np.linspace(0, 2*np.pi, np.random.randint(*range2d), endpoint=False))
  return pg

def gen_random_geometry_fanflat_vec():
  Vectors = np.zeros([16,6])
  # We assume constant detector width in these tests
  if not NONUNITDET:
    w = 1.0
  else:
    w = 0.6 + 0.8 * np.random.random()
  for i in range(Vectors.shape[0]):
    angle1 = 2*np.pi*np.random.random()
    if OBLIQUE:
      angle2 = angle1 + 0.5 * np.random.random()
    else:
      angle2 = angle1
    dist1 = 256 * (0.5 + np.random.random())
    detc = 10 * np.random.random(size=2)
    detu = [ math.cos(angle1) * w, math.sin(angle1) * w ]
    src = [ math.sin(angle2) * dist1, -math.cos(angle2) * dist1 ]
    Vectors[i, :] = [ src[0], src[1], detc[0], detc[1], detu[0], detu[1] ]
  pg = astra.create_proj_geom('fanflat_vec', np.random.randint(*range2d), Vectors)
  return pg

def gen_random_geometry_parallel_vec():
  Vectors = np.zeros([16,6])
  # We assume constant detector width in these tests
  if not NONUNITDET:
    w = 1.0
  else:
    w = 0.6 + 0.8 * np.random.random()
  for i in range(Vectors.shape[0]):
    l = 0.6 + 0.8 * np.random.random()
    angle1 = 2*np.pi*np.random.random()
    if OBLIQUE:
      angle2 = angle1 + 0.5 * np.random.random()
    else:
      angle2 = angle1
    detc = 10 * np.random.random(size=2)
    detu = [ math.cos(angle1) * w, math.sin(angle1) * w ]
    ray = [ math.sin(angle2) * l, -math.cos(angle2) * l ]
    Vectors[i, :] = [ ray[0], ray[1], detc[0], detc[1], detu[0], detu[1] ]
  pg = astra.create_proj_geom('parallel_vec', np.random.randint(*range2d), Vectors)
  return pg




def proj_type_to_fan(t):
  if t == 'cuda':
    return t
  else:
    return t + '_fanflat'

def display_mismatch(data, sinogram, a):
  pylab.gray()
  pylab.imshow(data)
  pylab.figure()
  pylab.imshow(sinogram)
  pylab.figure()
  pylab.imshow(a)
  pylab.figure()
  pylab.imshow(sinogram-a)
  pylab.show()

def display_mismatch_triple(data, sinogram, a, b, c):
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

class Test2DKernel(unittest.TestCase):
  def single_test(self, type, proj_type):
      shape = np.random.randint(*range2d, size=2)
      # these rectangles are biased, but that shouldn't matter
      rect_min = [ np.random.randint(0, a) for a in shape ]
      rect_max = [ np.random.randint(rect_min[i]+1, shape[i]+1) for i in range(len(shape))]
      if FLEXVOL:
          if not NONSQUARE:
            pixsize = np.array([0.5, 0.5]) + np.random.random()
          else:
            pixsize = 0.5 + np.random.random(size=2)
          origin = 10 * np.random.random(size=2)
      else:
          pixsize = (1.,1.)
          origin = (0.,0.)
      vg = astra.create_vol_geom(shape[1], shape[0],
                                 origin[0] - 0.5 * shape[0] * pixsize[0],
                                 origin[0] + 0.5 * shape[0] * pixsize[0],
                                 origin[1] - 0.5 * shape[1] * pixsize[1],
                                 origin[1] + 0.5 * shape[1] * pixsize[1])

      if type == 'parallel':
        pg = gen_random_geometry_parallel()
        projector_id = astra.create_projector(proj_type, pg, vg)
      elif type == 'parallel_vec':
        pg = gen_random_geometry_parallel_vec()
        projector_id = astra.create_projector(proj_type, pg, vg)
      elif type == 'fanflat':
        pg = gen_random_geometry_fanflat()
        projector_id = astra.create_projector(proj_type_to_fan(proj_type), pg, vg)
      elif type == 'fanflat_vec':
        pg = gen_random_geometry_fanflat_vec()
        projector_id = astra.create_projector(proj_type_to_fan(proj_type), pg, vg)


      data = np.zeros((shape[1], shape[0]), dtype=np.float32)
      data[rect_min[1]:rect_max[1],rect_min[0]:rect_max[0]] = 1

      sinogram_id, sinogram = astra.create_sino(data, projector_id)

      self.assertTrue(np.all(np.isfinite(sinogram)))

      #print(pg)
      #print(vg)

      astra.data2d.delete(sinogram_id)

      astra.projector.delete(projector_id)

      # NB: Flipped y-axis here, since that is how astra interprets 2D volumes
      xmin = origin[0] + (-0.5 * shape[0] + rect_min[0]) * pixsize[0]
      xmax = origin[0] + (-0.5 * shape[0] + rect_max[0]) * pixsize[0]
      ymin = origin[1] + (+0.5 * shape[1] - rect_max[1]) * pixsize[1]
      ymax = origin[1] + (+0.5 * shape[1] - rect_min[1]) * pixsize[1]

      if proj_type == 'line':

        a = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        b = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        c = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)

        for i, (center, edge1, edge2) in enumerate(gen_lines(pg)):
          (src, det) = center

          try:
            detweight = pg['DetectorWidth']
          except KeyError:
            if 'fan' not in type:
              detweight = effective_detweight(src, det, pg['Vectors'][i//pg['DetectorCount'],4:6])
            else:
              detweight = np.linalg.norm(pg['Vectors'][i//pg['DetectorCount'],4:6], ord=2)

          # We compute line intersections with slightly bigger (cw) and
          # smaller (aw) rectangles, and see if the kernel falls
          # between these two values.
          (aw,bw,cw) = intersect_line_rectangle_interval(src, det,
                        xmin, xmax, ymin, ymax,
                        1e-3)
          a[i] = aw * detweight
          b[i] = bw * detweight
          c[i] = cw * detweight
        a = a.reshape(astra.functions.geom_size(pg))
        b = b.reshape(astra.functions.geom_size(pg))
        c = c.reshape(astra.functions.geom_size(pg))

        if not np.all(np.isfinite(a)):
          raise RuntimeError("Invalid value in reference sinogram")
        if not np.all(np.isfinite(b)):
          raise RuntimeError("Invalid value in reference sinogram")
        if not np.all(np.isfinite(c)):
          raise RuntimeError("Invalid value in reference sinogram")
        self.assertTrue(np.all(np.isfinite(sinogram)))

        # Check if sinogram lies between a and c
        y = np.min(sinogram-a)
        z = np.min(c-sinogram)
        if DISPLAY and (z < 0 or y < 0):
          display_mismatch_triple(data, sinogram, a, b, c)
        self.assertFalse(z < 0 or y < 0)
      elif proj_type == 'linear' or proj_type == 'cuda':
        a = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        for i, (center, edge1, edge2) in enumerate(gen_lines(pg)):
          (src, det) = center
          (xd, yd) = det - src
          try:
            detweight = pg['DetectorWidth']
          except KeyError:
            if 'fan' not in type:
              detweight = effective_detweight(src, det, pg['Vectors'][i//pg['DetectorCount'],4:6])
            else:
              detweight = np.linalg.norm(pg['Vectors'][i//pg['DetectorCount'],4:6], ord=2)

          l = 0.0
          if np.abs(xd) > np.abs(yd): # horizontal ray
            length = math.sqrt(1.0 + abs(yd/xd)**2)
            y_seg = (ymin, ymax)
            for j in range(rect_min[0], rect_max[0]):
              x = origin[0] + (-0.5 * shape[0] + j + 0.5) * pixsize[0]
              w = intersect_line_vertical_segment_linear(center[0], center[1], x, y_seg, pixsize[1])
              # limited interpolation precision with cuda
              if CUDA_8BIT_LINEAR and proj_type == 'cuda':
                w = np.round(w * 256.0) / 256.0
              l += w * length * pixsize[0] * detweight
          else:
            length = math.sqrt(1.0 + abs(xd/yd)**2)
            x_seg = (xmin, xmax)
            for j in range(rect_min[1], rect_max[1]):
              y = origin[1] + (+0.5 * shape[1] - j - 0.5) * pixsize[1]
              w = intersect_line_horizontal_segment_linear(center[0], center[1], y, x_seg, pixsize[0])
              # limited interpolation precision with cuda
              if CUDA_8BIT_LINEAR and proj_type == 'cuda':
                w = np.round(w * 256.0) / 256.0
              l += w * length * pixsize[1] * detweight
          a[i] = l
        a = a.reshape(astra.functions.geom_size(pg))
        if not np.all(np.isfinite(a)):
          raise RuntimeError("Invalid value in reference sinogram")
        x = np.max(np.abs(sinogram-a))
        TOL = 2e-3 if proj_type != 'cuda' else CUDA_TOL
        if DISPLAY and x > TOL:
          display_mismatch(data, sinogram, a)
        self.assertFalse(x > TOL)
      elif proj_type == 'distance_driven':
        a = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        for i, (center, edge1, edge2) in enumerate(gen_lines(pg)):
          (xd, yd) = center[1] - center[0]
          l = 0.0
          if np.abs(xd) > np.abs(yd): # horizontal ray
            y_seg = (ymin, ymax)
            for j in range(rect_min[0], rect_max[0]):
              x = origin[0] + (-0.5 * shape[0] + j + 0.5) * pixsize[0]
              l += intersect_ray_vertical_segment(edge1, edge2, x, y_seg) * pixsize[0]
          else:
            x_seg = (xmin, xmax)
            for j in range(rect_min[1], rect_max[1]):
              y = origin[1] + (+0.5 * shape[1] - j - 0.5) * pixsize[1]
              l += intersect_ray_horizontal_segment(edge1, edge2, y, x_seg) * pixsize[1]
          a[i] = l
        a = a.reshape(astra.functions.geom_size(pg))
        if not np.all(np.isfinite(a)):
          raise RuntimeError("Invalid value in reference sinogram")
        x = np.max(np.abs(sinogram-a))
        TOL = 2e-3
        if DISPLAY and x > TOL:
          display_mismatch(data, sinogram, a)
        self.assertFalse(x > TOL)
      elif proj_type == 'strip' and 'fan' in type:
        a = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        for i, (center, edge1, edge2) in enumerate(gen_lines(pg)):
          (src, det) = center
          det_dist = np.linalg.norm(src-det, ord=2)
          l = 0.0
          for j in range(rect_min[0], rect_max[0]):
            xmin = origin[0] + (-0.5 * shape[0] + j) * pixsize[0]
            xmax = origin[0] + (-0.5 * shape[0] + j + 1) * pixsize[0]
            xcen = 0.5 * (xmin + xmax)
            for k in range(rect_min[1], rect_max[1]):
              ymin = origin[1] + (+0.5 * shape[1] - k - 1) * pixsize[1]
              ymax = origin[1] + (+0.5 * shape[1] - k) * pixsize[1]
              ycen = 0.5 * (ymin + ymax)
              scale = det_dist / np.linalg.norm( src - np.array((xcen,ycen)), ord=2 )
              w = intersect_ray_rect(edge1, edge2, xmin, xmax, ymin, ymax)
              l += w * scale
          a[i] = l
        a = a.reshape(astra.functions.geom_size(pg))
        if not np.all(np.isfinite(a)):
          raise RuntimeError("Invalid value in reference sinogram")
        x = np.max(np.abs(sinogram-a))
        TOL = 8e-3
        if DISPLAY and x > TOL:
          display_mismatch(data, sinogram, a)
        self.assertFalse(x > TOL)
      elif proj_type == 'strip':
        a = np.zeros(np.prod(astra.functions.geom_size(pg)), dtype=np.float32)
        for i, (center, edge1, edge2) in enumerate(gen_lines(pg)):
          a[i] = intersect_ray_rect(edge1, edge2, xmin, xmax, ymin, ymax)
        a = a.reshape(astra.functions.geom_size(pg))
        if not np.all(np.isfinite(a)):
          raise RuntimeError("Invalid value in reference sinogram")
        x = np.max(np.abs(sinogram-a))
        TOL = 8e-3
        if DISPLAY and x > TOL:
          display_mismatch(data, sinogram, a)
        self.assertFalse(x > TOL)

  def multi_test(self, type, proj_type):
    np.random.seed(seed)
    for _ in range(nloops):
      self.single_test(type, proj_type)

  def test_par(self):
    self.multi_test('parallel', 'line')
  def test_par_linear(self):
    self.multi_test('parallel', 'linear')
  def test_par_cuda(self):
    self.multi_test('parallel', 'cuda')
  def test_par_dd(self):
    self.multi_test('parallel', 'distance_driven')
  def test_par_strip(self):
    self.multi_test('parallel', 'strip')
  def test_fan(self):
    self.multi_test('fanflat', 'line')
  def test_fan_strip(self):
    self.multi_test('fanflat', 'strip')
  def test_fan_cuda(self):
    self.multi_test('fanflat', 'cuda')
  def test_parvec(self):
    self.multi_test('parallel_vec', 'line')
  def test_parvec_linear(self):
    self.multi_test('parallel_vec', 'linear')
  def test_parvec_dd(self):
    self.multi_test('parallel_vec', 'distance_driven')
  def test_parvec_strip(self):
    self.multi_test('parallel_vec', 'strip')
  def test_parvec_cuda(self):
    self.multi_test('parallel_vec', 'cuda')
  def test_fanvec(self):
    self.multi_test('fanflat_vec', 'line')
  def test_fanvec_cuda(self):
    self.multi_test('fanflat_vec', 'cuda')





if __name__ == '__main__':
  unittest.main()

