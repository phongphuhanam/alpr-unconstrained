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

# input_dir  = sys.argv[1]
def main(input_dir, net_path):
    input_dir = "/media/data1/datasets/CJ/s_geq_1_d_leq_20/0000_03/"
    output_dir = "./output"

    lp_threshold = .5

    # wpod_net_path = sys.argv[2]
    wpod_net = load_model(net_path)
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
            	Ilp = LlpImgs[0]
            	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                s = Shape(Llp[0].pts)
                cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
                writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
            # print current_frame.head()
            # 
            # Ivehicle = cv2.imread(img_path)
            # for a in current_frame
            # ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            # 
            # 
            # print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

            # Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

            # if len(LlpImgs):
            # 	Ilp = LlpImgs[0]
            # 	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            # 	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            # 	s = Shape(Llp[0].pts)

            # 	cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
            # 	writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

    # except:
    #     traceback.print_exc()
    #     sys.exit(1)

    # sys.exit(0)
    # -

if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])
    except:
        traceback.print_exc()
    sys.exit(0)

