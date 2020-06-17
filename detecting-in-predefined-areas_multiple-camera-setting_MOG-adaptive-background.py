########################################################################################################################
# May 2019
# Jennifer Salau, Institute of animal breeding and husbandry, Kiel University
# jsalau@tierzucht.uni-kiel.de
#
# USE AT OWN RISK
########################################################################################################################
import numpy as np
import pandas as pd
import cv2
import os


####################FUNCTIONS####################
#
def intersection(binary_img, cutout):
    """
    Jennifer Salau, 13th of May 2019
    Takes
    -binary_img: binary image
    -cutout: a list of four floats in [0,1] to define the cutout limits (height / width) given in percentages of image height / width
    #
    Returns bool: Are there are nonzero pixels in the given cutout.
    """
    h = binary_img.shape[0]
    w = binary_img.shape[1]
    b = np.any(binary_img[int(cutout[0] * h):int(cutout[1] * h), int(cutout[2] * w):int(cutout[3] * w)])
    return (b)
#
def detect_movement_in_predefined_areas(binary_img, areas_dict):
    """
    Jennifer Salau, 13th of May 2019
    Takes
    -binary_img: binary image (non zero pixels indicating movement in the original image, i.e. the result of a search for connected components
    -areas_dict: predefined areas as dictionary {key_area1_id: list of 4 floats in [0,1] to define the limits of area1 in percentages of height and width of the image, key_area2_id:[...],...}
    Returns a one-row pandas DataFrame with the area IDs as column names and boolean entries according to non zero pixels in the respective areas.
    """
    areas_list = [v for v in areas_dict.values()]
    bool_list = [intersection(binary_img, a) for a in areas_list]
    areas_names = [v for v in areas_dict.keys()]
    df = pd.DataFrame(data=dict(zip(areas_names, bool_list)), index=np.arange(1))

    return (df)
#
def tidying_connected_components(conn_comps, px_sz, **kwargs):
    """
    Jennifer Salau, 13th of May 2019
    Required arguments:
    -conn_comps: the result of a search for connected components (cv2.connectedComponents() / cv2.connectedComponentsWithStats() )
    -px_sz: an integer to specify the minimal number of pixels for a connected component to be kept
    Keyword arguments:
    -stats: the stats ouptput, if conn_comps was derived from cv2.connectedComponentsWithStats()
    -centroids: the centroids ouptput, if conn_comps was derived from cv2.connectedComponentsWithStats()
    Returns a binary image with only the connected components with at least the given number of pixels.
    """
    unique, counts = np.unique(conn_comps, return_counts=True)
    X = np.asarray((unique, counts)).T
    #
    to_del = [np.transpose(np.nonzero(conn_comps == x[0])) for x in X if x[1] < px_sz]
    to_del_indices = np.asarray([[ll[0], ll[1]] for l in to_del for ll in l]).T
    #
    for i in range(len(to_del_indices[0])):
        conn_comps[to_del_indices[0][i], to_del_indices[1][i]] = 0

    conn_comps[np.where(conn_comps > 0)] = 255
    conn_comps = 255 * conn_comps / np.amax(conn_comps)  
    if kwargs:
        stats = kwargs["stats"]
        centroids = kwargs["centroids"]
        #
        centroids = centroids[np.nonzero(stats[:, -1] >= px_sz), :]
        stats = stats[np.nonzero(stats[:, -1] >= px_sz), :]
        #
        return ((conn_comps, stats, centroids))
    else:
        return (conn_comps)
#
def motiondetection(videoname, ppath, px_sz, inner_areas, CC_stats=True):
    """
    Jennifer Salau, 13th of May 2019
    Required arguments:
    -videoname: the name of the mp4-video file to process WITHOUT the file ending
    -ppath: the absolute path to the video file
    -px_sz: an integer to specify the minimal number of pixels for a connected component to be kept
    -inner_areas: dictionary holding the boundaries of the inner regions of the user defined barn segments with regard to the field of view of the respective camera, e.g. 
		inner_areas =  {'name_of_first_segment': [0.25, 0.585, 0.085, 0.11],...,'name_of_last_segment': [0.775, 1, 0.72, 0.88]}, whereas
		[startY, endY, startX, endX] are percentages of the total height (Y) or width (X) starting at the upper left corner of the image^^

    -CC_stats: If True (default) 'cv2.connectedComponentsWithStats' is applied and 'cv2.connectedComponents' otherwise.
    Returns a data frame containing one row for each processed frame and one column for every segment defined in inner_areas; entries are True or False whether the respective segment was occupied by a cow or not. This data frame is, additionally, written to csv file.
    """
    # open given video file:
    cap = cv2.VideoCapture(os.path.join(ppath, f'{videoname}.mp4'))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    # create MOG (mixture of gaussians) background subtractor object
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    # pre-initiate DataFrame for movement detection:
    df_movement = pd.DataFrame()
    # start looping
    count = 0
    ret = True
    while ret:
        ret, frame = cap.read()
#####
# Define a condition to which frames of the video the motion detection should be applied; here it is the first frame of every minute as our videos have been recorded with two frames per second!
#######
        if (count %  120 == 0): 
            #print(count)
            fgmask = fgbg.apply(frame)
            if np.any(fgmask):
                if CC_stats:
                    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
                    # stats :  statistics output for each label, including the background label, see below for available statistics.Statistics are accessed via stats(label, COLUMN) where COLUMN is one of ConnectedComponentsTypes.The data type is CV_32S.
                    #   cv.CC_STAT_LEFT -> The leftmost(x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
                    #   cv.CC_STAT_TOP -> The topmost(y) coordinate which is the inclusive start of the bounding box in the vertical direction.
                    #   cv.CC_STAT_WIDTH -> The horizontal size of the bounding box.
                    #   cv.CC_STAT_HEIGHT -> The vertical size of the bounding box.
                    #   cv.CC_STAT_AREA -> The total area( in pixels) of the connected component.
                    #   cv.CC_STAT_MAX
                    # centroids : centroid output for each label, including the background label.Centroids are accessed via centroids(label, 0) for x and centroids(label, 1) for y.The data type CV_64F.
                    conn_comps, stats, centroids = tidying_connected_components(labels, px_sz, stats=stats, centroids=centroids)
                else:
                    ret, labels, = cv2.connectedComponents(fgmask)
                    conn_comps = tidying_connected_components(labels, px_sz)

                df = detect_movement_in_predefined_areas(labels, inner_areas)
                df_movement = df_movement.append(df)
        count = count + 1

    cap.release()
    cv2.destroyAllWindows()
    df_movement.to_csv(os.path.join(ppath, f'MOVEMENT_{videoname}_{px_sz}.csv'))
    return(df_movement)
#
def cumulate_cow_presence(df, chunksize):
    """
    Jennifer Salau, 27th of June 2019
    Required arguments
    -df: a pd.DataFrame holding one row for each area associated with the respective camera holding True/False (1/0)
        entries depending on whether motion was detected in that area or not (one row for each tested image)
    -chunksize: an integer, specifying the size of the chunks over which to accumulate detected motion
    #
    Returns a pd.DataFrame with the same column names as input 'df'; holding the counts of detected motion in the areas
        cumulated for chunks of rows/images specified by input 'chunksize'
    """
    cow_presence_df = pd.DataFrame()
    chunk_mod = len(df.index) % chunksize
    if chunk_mod == 0:
        chunks = [np.arange(i, i + chunksize) for i in np.arange(0, len(df.index), chunksize)]
    elif chunk_mod <= 0.5*chunksize:
        chunks = [np.arange(i, i + chunksize) for i in np.arange(0, len(df.index)-chunksize-chunk_mod, chunksize)] + [np.arange(len(df.index)-chunksize-chunk_mod, len(df.index))]
    else:
        chunks = [np.arange(i, i + chunksize) for i in np.arange(0, len(df.index) - chunk_mod, chunksize)] + [np.arange(len(df.index) - chunk_mod, len(df.index))]
    #
    for c in chunks:
        cow_presence_df = cow_presence_df.append(df.iloc[c,:].sum(axis = 0),ignore_index=True)

    return(cow_presence_df)
#
#
####################PARAMETER####################
#
########################################################################################################################
# Define the inner_areas for all cameras of your installation here:
########################################################################################################################
inner_areas_1stcam =  {'name_of_first_segment_1stcam': [startY, endY, startX, endX],...,'name_of_last_segment_1stcam': [startY, endY, startX, endX]}
...
inner_areas_lastcam =  {'name_of_first_segment_lastcam': [startY, endY, startX, endX],...,'name_of_last_segment_lastcam': [startY, endY, startX, endX]}
#
########################################################################################################################
# Put the videos you want to process into the same folder (i.e. VIDEOS2PROCESS) and set ppath to the absolute path to this 
# video files
########################################################################################################################
ppath = os.path.join(os.path.realpath('..'), 'VIDEOS2PROCESS')
#

########################################################################################################################
# Define the minimal size (integer values) a connected component must have to NOT be considered background noise and get 
# deleted. All connected components with at least px pixels will be kept as foreground objects.
# Store the px values in a list with length equal to the numbers of cameras used in your installation!
#########################################################################################################################
P = [px_1stcam, ..., px_lastcam]
#
########################################################################################################################
# Define the chunksize in minutes to accumulate the cow presence
########################################################################################################################
chunksizes = [30, 60, 180]
#
########################################################################################################################
# Construct a list containing one dictionary for every video that should be processed. keys needed are 'videoname', 'px_sz'
# (fill in the entry of P corresponding to the camera with which the video under 'videoname' was recorded), and 'inner_areas' 
########################################################################################################################
D =  [{'videoname': file_name_1stvideo_1stcam, 'px_sz': P[0] , 'inner_areas': inner_areas_1stcam},
...
{'videoname': file_name_1stvideo_lastcam, 'px_sz': P[-1] , 'inner_areas': inner_areas_lastcam}
...
...
file_name_lastvideo_1stcam, 'px_sz': P[0] , 'inner_areas': inner_areas_1stcam},
...
{'videoname': file_name_lastvideo_lastcam, 'px_sz': P[-1] , 'inner_areas': inner_areas_lastcam}
]
#
#
########################################################################################################################
# In the following loop all videos in D will be processed using 'motiondetection'. The results will be stored as csv file
# and cumulated according to the user given chunksize. The cumulation results no longer contain binary entries, but sums of
# cow presences within the given chunks, and will also be stored as csv files.
#
# Both types of csv files were written for every video separately! For an analysis of cow presence in your complete area of
# observation the resulting data frames of simultaneously recorded videos of all cameras covering your barn have to be 
# merged according to your needs
########################################################################################################################
if __name__ == '__main__':
	for d in D:
		videoname = d['videoname']
		print("Processing video: " + videoname)
		px_sz = d['px_sz']
		df = motiondetection(videoname, ppath, px_sz, d['inner_areas'], False)
		if not os.path.exists(os.path.join(ppath, 'Results')):
			 os.mkdir(os.path.join(ppath, 'Results'))
		df.to_csv(os.path.join(ppath, 'Results', f'MotionDetection_{videoname}-{px_sz}.csv'))
		#
		for c in chunksizes:
			cow_presence_df = cumulate_cow_presence(df, c)
			cow_presence_df.to_csv(os.path.join(ppath, 'Results', f'CowPresence_{videoname}-{px_sz}-Chunksize{c}.csv'))
		#
		cow_presence_df = cumulate_cow_presence(df, len(df.index))
		cow_presence_df.to_csv(os.path.join(ppath, 'Results', f'CowPresence_{videoname}-{px_sz}-WholeDay.csv'))
		print("done.")
#
########################################################################################################################
