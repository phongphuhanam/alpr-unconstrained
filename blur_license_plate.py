import sys, os
import keras
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
import pandas as pd
import numpy as np

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

def blurring(image, factor=3.0):
    (h,w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h /factor)
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1
    return cv2.GaussianBlur(image, (kW, kH), 0)

def draw_losangle(I,pts,color=(1.,1.,1.),thickness=1):
	assert(pts.shape[0] == 2 and pts.shape[1] == 4)

	for i in range(4):
		pt1 = tuple(pts[:,i].astype(int).tolist())
		pt2 = tuple(pts[:,(i+1)%4].astype(int).tolist())
		cv2.line(I,pt1,pt2,color,thickness)

# input_dir  = sys.argv[1]
def main(input_dir, wpod_net):
    # input_dir = "/media/data1/datasets/CJ/s_geq_1_d_leq_20/0000_03/"
    output_dir = "./output"

    lp_threshold = .5

    # wpod_net_path = sys.argv[2]
    
    gt_path = os.path.join(input_dir, "gt", "gt.txt")
    # tracklets_names = ['FrameID', 'trackid', 'top', 'left', 'width', 'height', 'prob', 'classid', 'reserverd1']
    gt_file = np.loadtxt(gt_path, delimiter=",").astype(np.int)
    # gt_file[:,4:6] += gt_file[:,2:4]
    # frame_ids = gt_file[:,0].astype(np.int)
    # save_df = pd.read_csv(gt_path_dir, names=tracklets_names, index_col=False)
    imgs_paths = sorted(glob(os.path.join(input_dir, "img1", "*.jpg")))

    print 'Searching for license plates using WPOD-NET'
    
    for i,img_path in enumerate(imgs_paths):
        print '\t Processing %s' % img_path
        bname = splitext(basename(img_path))[0]
        frame_id = int(bname)
        current_frame = gt_file[gt_file[:,0] == frame_id] 
        # print(current_frame)
        ivehicle = cv2.imread(img_path, cv2.IMREAD_COLOR)

        def draw_losangle(pts,color=(0,0,255),thickness=1):
            assert(pts.shape[0] == 2 and pts.shape[1] == 4)
            for i in range(4):
                pt1 = tuple(pts[:,i].astype(int).tolist())
                pt2 = tuple(pts[:,(i+1)%4].astype(int).tolist())
                cv2.line(ivehicle,pt1,pt2,color,thickness)

        im_wh = np.array(ivehicle.shape[1::-1], dtype=float)
        is_lp_found = False
        for a in current_frame:
            br = a[2:4] + a[4:6]
            veh_img = ivehicle[a[3]:br[1],a[2]:br[0]]
            output_path = os.path.join("output", "%s_%02d.jpg" % (bname, a[1]))
            # cv2.imwrite(output_path, veh_img)
            ratio = float(max(a[4:6]))/min(a[4:6])
            side  = int(ratio*288.)
            bound_dim = min(side + (side%(2**4)),608)
            # print(a)
            Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(veh_img),bound_dim,2**4,(240,80),lp_threshold)

            if len(LlpImgs):
                wh_ratio = a[4:6] / im_wh
                tl_ratio = a[2:4] / im_wh
            	Ilp = LlpImgs[0]
            	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                s = Shape(Llp[0].pts)
                img_pts = Llp[0].pts*wh_ratio.reshape(2,1) + tl_ratio.reshape(2,1)
                abs_pts = img_pts*im_wh.reshape(2,1)
                abs_pts_2 = abs_pts.T.astype(np.int)
                cv2.fillPoly(ivehicle, [abs_pts_2], (213, 214,206))
                # draw_losangle(abs_pts)
                is_lp_found = True
        
        if is_lp_found:
            cv2.imwrite(img_path,ivehicle)
    

if __name__ == '__main__':
    net_path = sys.argv[2]
    wpod_net = load_model(net_path)
    try:
        for a in glob(os.path.join(sys.argv[1], "*")):
            if os.path.isdir(os.path.join(a, "img1")):
                main(a, wpod_net)
    except:
        traceback.print_exc()
    sys.exit(0)

