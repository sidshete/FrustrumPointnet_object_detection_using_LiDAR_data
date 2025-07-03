import numpy as np

class Calibration:
    def __init__(self, calib_file):
        self.P, self.V2C, self.R0 = self.read_calib_file(calib_file)

    def read_calib_file(self, calib_file):
        with open(calib_file, 'r') as f:
            lines = f.readlines()

        calibration = {}
        for line in lines:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                calibration[key] = np.array([float(i) for i in value.split()])

        V2C = calibration.get('Tr_velo_to_cam', np.eye(4))
        if V2C.shape == (12,):
            V2C = V2C.reshape(3, 4)

        R0 = calibration.get('R0_rect', np.eye(3))
        if R0.shape == (9,):
            R0 = R0.reshape(3, 3)

        P0 = calibration.get('P0', np.zeros((3, 4)))  # <- Default to zero matrix
        if P0.shape == (12,):
            P0 = P0.reshape(3, 4)
            # print("Shape of P:", P0.shape)
            # print("Shape of V2C:", V2C.shape)
            # print("Shape of R0:", R0.shape)


        return P0, V2C, R0


    def project_velo_to_cam(self, pts_3d_velo):
        if pts_3d_velo.shape[1] == 3:
            pts_3d_velo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))
        pts_3d_cam = np.dot(self.V2C, pts_3d_velo.T).T
        pts_3d_cam = np.dot(self.R0, pts_3d_cam.T).T
        return pts_3d_cam

    def project_cam_to_image(self, pts_3d_cam):
        pts_3d_cam = np.insert(pts_3d_cam, 3, 1, axis=1)
        pts_2d = np.dot(self.P, pts_3d_cam.T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, :2]

    def get_frustum(self, box2d, pc_velo):
        pts_cam = self.project_velo_to_cam(pc_velo)
        pts_img = self.project_cam_to_image(pts_cam)
        x1, y1, x2, y2 = box2d
        mask = (
            (pts_img[:, 0] >= x1) & (pts_img[:, 0] <= x2) &
            (pts_img[:, 1] >= y1) & (pts_img[:, 1] <= y2) &
            (pts_cam[:, 2] > 0)
        )
        return pc_velo[mask]
