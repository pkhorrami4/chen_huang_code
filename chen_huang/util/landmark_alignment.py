import numpy
from skimage.transform import AffineTransform, matrix_transform


def align_pts_to_template(pts, template_pts, display=False):
    pts_r = numpy.reshape(pts, (2, -1), 'F')
    template_pts_r = numpy.reshape(template_pts, (2, -1), 'F')
    # print pts_r

    nose_pts = pts_r[:, 27:36]
    eye_right_pts = pts_r[:, 36:42]
    eye_left_pts = pts_r[:, 42:48]
    mouth_pts = pts_r[:, 48:]
    # print 'Nose Pts: ', nose_pts
    # print 'Right Eye Pts: ', eye_right_pts
    # print 'Left Eye Pts: ', eye_left_pts

    nose_template_pts = template_pts_r[:, 27:36]
    eye_right_template_pts = template_pts_r[:, 36:42]
    eye_left_template_pts = template_pts_r[:, 42:48]
    # print 'Nose Pts (mean): ', nose_template_pts
    # print 'Right Eye Pts (mean): ', eye_right_template_pts
    # print 'Left Eye Pts (mean): ', eye_left_template_pts

    src_pts = numpy.zeros((13, 2), dtype='float32')
    src_pts[0:9, :] = nose_pts.T
    src_pts[9, :] = eye_right_pts[:, 0].T
    src_pts[10, :] = eye_right_pts[:, 3].T
    src_pts[11, :] = eye_left_pts[:, 0].T
    src_pts[12, :] = eye_left_pts[:, 3].T

    dest_pts = numpy.zeros((13, 2), dtype='float32')
    dest_pts[0:9, :] = nose_template_pts.T
    dest_pts[9, :] = eye_right_template_pts[:, 0].T
    dest_pts[10, :] = eye_right_template_pts[:, 3].T
    dest_pts[11, :] = eye_left_template_pts[:, 0].T
    dest_pts[12, :] = eye_left_template_pts[:, 3].T

    # src_pts = numpy.zeros((3, 2), dtype='float32')
    # src_pts[0, :] = pts_r[:, 33].T
    # src_pts[1, :] = pts_r[:, 36].T
    # src_pts[2, :] = pts_r[:, 45].T

    # dest_pts = numpy.zeros((3, 2), dtype='float32')
    # dest_pts[0, :] = template_pts_r[:, 33].T
    # dest_pts[1, :] = template_pts_r[:, 36].T
    # dest_pts[2, :] = template_pts_r[:, 45].T

    tform = AffineTransform()
    tform.estimate(src_pts, dest_pts)
    # print tform.params

    pts_transformed_r = matrix_transform(pts_r.T, tform.params)
    # print pts_transformed

    if display:
        plt.scatter(pts_r[0, :], -pts_r[1, :])
        plt.scatter(pts_r[0, 48:60], -pts_r[1, 48:60], c='r')
        plt.scatter(pts_r[0, 60:], -pts_r[1, 60:], c='g')
        plt.title('Pts Before Alignment')
        plt.show()

        plt.scatter(pts_transformed_r[:, 0], -pts_transformed_r[:, 1])
        plt.title('Pts After Alignment')
        plt.show()

    pts_transformed = numpy.reshape(pts_transformed_r, -1)
    return pts_transformed
