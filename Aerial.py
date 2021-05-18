#requirements
import pandas as pd
import geemap
import os
import ee
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import argparse
import imutils
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from vision import Vision

class Fire(Vision):
        def __init__(self):
            print('Fire obj created...')
            
        def transform(self, read_from_path, write_to_path):
                
                img = cv2.imread(read_from_path)
 
                lower_bound = np.array([5,50,100],np.uint8)
                upper_bound = np.array([15,255,255],np.uint8)
                 
                frame_smooth = cv2.GaussianBlur(img,(15,15),0)
                 
                mask = np.zeros_like(img)
                   
                mask[0:img.shape[0], 0:img.shape[1]] = [255,255,255]
                 
                img_roi = cv2.bitwise_and(frame_smooth, mask)
                 
                frame_hsv = cv2.cvtColor(img_roi,cv2.COLOR_BGR2HSV)
                 
                image_binary = cv2.inRange(frame_hsv, lower_bound, upper_bound)

                cv2.imwrite(write_to_path, image_binary)

def generate_detections(checkpoint,images):
                    
                    print("Creating Graph...")
                    detection_graph = tf.Graph()
                    with detection_graph.as_default():
                        od_graph_def = tf.GraphDef()
                        with tf.gfile.GFile(checkpoint, 'rb') as fid:
                            serialized_graph = fid.read()
                            od_graph_def.ParseFromString(serialized_graph)
                            tf.import_graph_def(od_graph_def, name='')

                    boxes = []
                    scores = []
                    classes = []
                    k = 0
                    with detection_graph.as_default():
                        with tf.Session(graph=detection_graph) as sess:
                            for image_np in tqdm(images):
                                image_np_expanded = np.expand_dims(image_np, axis=0)
                                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                                score = detection_graph.get_tensor_by_name('detection_scores:0')
                                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                                # Actual detection.
                                (box, score, clss, num_detections) = sess.run(
                                        [box, score, clss, num_detections],
                                        feed_dict={image_tensor: image_np_expanded})

                                boxes.append(box)
                                scores.append(score)
                                classes.append(clss)
                                
                    boxes =   np.squeeze(np.array(boxes))
                    scores = np.squeeze(np.array(scores))
                    classes = np.squeeze(np.array(classes))

                    return boxes,scores,classes

def chip_image(img, chip_size=(60,60)):

                    width,height,_ = img.shape
                    wn,hn = chip_size
                    images = np.zeros((int(width/wn) * int(height/hn),wn,hn,3))
                    k = 0
                    for i in tqdm(range(int(width/wn))):
                        for j in range(int(height/hn)):
                            
                            chip = img[wn*i:wn*(i+1),hn*j:hn*(j+1),:3]
                            images[k]=chip
                            
                            k = k + 1
                    
                    return images.astype(np.uint8)

def draw_bboxes(img,boxes,classes):

                    source = Image.fromarray(img)
                    draw = ImageDraw.Draw(source)
                    w2,h2 = (img.shape[0],img.shape[1])

                    idx = 0

                    for i in range(len(boxes)):
                        xmin,ymin,xmax,ymax = boxes[i]
                        c = classes[i]

                        draw.text((xmin+15,ymin+15), str(c))

                        for j in range(4):
                            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
                    return source



class Building(Vision):
        def __init__(self):
                self.chip_size = 60
                self.path_to_model='multires_aug.pb'
                print('Building obj created...')
            
        def transform(self, read_from_path, write_to_path):
   
                #Parse and chip images
                arr = np.array(Image.open(read_from_path))
                chip_size = (self.chip_size,self.chip_size)
                images = chip_image(arr,chip_size)
                print(images.shape)

                #generate detections
                boxes, scores, classes = generate_detections(self.path_to_model,images)

                #Process boxes to be full-sized
                width,height,_ = arr.shape
                cwn,chn = (chip_size)
                wn,hn = (int(width/cwn),int(height/chn))

                num_preds = 250
                bfull = boxes[:wn*hn].reshape((wn,hn,num_preds,4))
                b2 = np.zeros(bfull.shape)
                b2[:,:,:,0] = bfull[:,:,:,1]
                b2[:,:,:,1] = bfull[:,:,:,0]
                b2[:,:,:,2] = bfull[:,:,:,3]
                b2[:,:,:,3] = bfull[:,:,:,2]

                bfull = b2
                bfull[:,:,:,0] *= cwn
                bfull[:,:,:,2] *= cwn
                bfull[:,:,:,1] *= chn
                bfull[:,:,:,3] *= chn
                for i in range(wn):
                    for j in range(hn):
                        bfull[i,j,:,0] += j*cwn
                        bfull[i,j,:,2] += j*cwn
                                    
                        bfull[i,j,:,1] += i*chn
                        bfull[i,j,:,3] += i*chn
                                    
                bfull = bfull.reshape((hn*wn,num_preds,4))

                #only display boxes with confidence > .2
                bs = bfull[scores > .2]
                cs = classes[scores>.2]
                s = (write_to_path).split("/")[::-1]
                draw_bboxes(arr,bs,cs).save(s[0].split(".")[0] + ".png")

                with open(write_to_path,'w') as f:
                    for i in range(bfull.shape[0]):
                        for j in range(bfull[i].shape[0]):
                            #box should be xmin ymin xmax ymax
                            box = bfull[i,j]
                            class_prediction = classes[i,j]
                            score_prediction = scores[i,j]
                            f.write('%d %d %d %d %d %f \n' % \
                                (box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))


#uploaded "testfire.jpg" and "satellite.tif" to possibly use for Demo unless you have other image to test. Note that we have two inputs and outputs because they're two tasks until we get sim aerial view with buildings and fire and/or get the Map interface connected.
#args: path to aerial fire image, path to output fire detection, path to aerial building image, path to output object/building detection
#arg: path to model from main Git page
#fire = Fire()
#building = Building()
#fire.transform("testfire.jpg", "firedetection.jpg")
#building.transform("satellite.tif", "satellite_objectdetection.txt")

#open pickle for current drone position, altitude, user input etc. (future work) 

class Path(Vision):
        def __init__(self):
                from scipy.stats import kde
                print('Path obj created...')

        def transform(self, read_from_path, write_to_path):
                img_raw = plt.imread(params['photos_path']+"0/satellite.tif")
                img_bb = plt.imread(params['photos_path']+"0/satellite.png")
                params = pickle.load(open( "params.p", "rb" ))
                photos_path = params['photos_path']
                preds = pd.read_csv(params['photos_path']+'0/satellite.txt', sep=' ', 
                  names=["Xmin", "Ymin", "Xmax", "Ymax", "Class ID", "Confidence"], index_col=False)
                t = preds.sort_values(by=['Confidence'], ascending=False)
                indexNames = t[t['Confidence'] < 0.2].index
                t.drop(indexNames , inplace=True)
                coords = t[["Xmin", "Ymin", "Xmax", "Ymax"]]
                x = []
                y = []
                centroid = []
                test = pd.DataFrame(coords).to_numpy()
                for i in test:
                    xmin = i[0]
                    ymin = i[1]
                    xmax = i[2]
                    ymax = i[3]
                    coord = (xmin, ymin, xmax, ymax)
                    centerCoord = (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))
                    centroid.append(centerCoord)
                    x.append(centerCoord[0])
                    y.append(centerCoord[1])
                import matplotlib.pyplot as plt
                import numpy as np
                import math
                x = []
                y = []
                for i in test:
                    xmin = i[0]
                    ymin = i[1]
                    xmax = i[2]
                    ymax = i[3]
                    x.append(xmin)
                    y.append(ymin)
                    x.append(xmax)
                    y.append(ymax)
                #DEFINE GRID SIZE AND RADIUS(h)
                grid_size=1
                h=30
                #GETTING X,Y MIN AND MAX
                x_min=0
                x_max=img_raw.shape[1]
                y_min=0
                y_max=img_raw.shape[0]
                #CONSTRUCT GRID
                x_grid=np.arange(x_min,x_max, grid_size)
                y_grid=np.arange(y_min,y_max, grid_size)
                x_mesh,y_mesh=np.meshgrid(x_grid,y_grid)
                #GRID CENTER POINT
                xc=x_mesh+(grid_size/2)
                yc=y_mesh+(grid_size/2)
                #FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
                def kde_quartic(d,h):
                    dn=d/h
                    P=(15/16)*(1-dn**2)**2
                    return P
                #PROCESSING
                intensity_list=[]
                for j in range(len(xc)):
                    intensity_row=[]
                    for k in range(len(xc[0])):
                        kde_value_list=[]
                        for i in range(len(x)):
                            #CALCULATE DISTANCE
                            d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2) 
                            if d<=h:
                                p=kde_quartic(d,h)
                            else:
                                p=0
                            kde_value_list.append(p)
                        #SUM ALL INTENSITY VALUE
                        p_total=sum(kde_value_list)
                        intensity_row.append(p_total)
                    intensity_list.append(intensity_row)
                #HEATMAP OUTPUT    
                intensity=np.array(intensity_list)
                fig = plt.figure()
                ax = fig.add_subplot(111) 
                plt.pcolormesh(x_mesh,y_mesh,intensity, cmap='plasma', shading='auto')
                #plt.xlim([0,bb.shape[1]]) 
                #plt.ylim([bb.shape[0],0])
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.savefig(photos_path + '0/density_map.png')

                maze = pd.DataFrame(intensity)
                maze = maze.astype(int)

                with open(photos_path + '0/coordinates.csv', 'r') as file:
                    data = file.read().replace('\n', ',').strip(',')

                res = data.strip(',').strip("'").split(',')
                res = [float(i) for i in res]
                bottomLeft = (res[1], res[0])
                bottomRight = (res[7], res[6])
                topLeft = (res[3], res[2])
                topRight = (res[5], res[4])
                cols = np.linspace(bottomLeft[1], bottomRight[1], num=len(maze.columns)+1)
                rows = np.linspace(bottomLeft[0], topLeft[0], num=len(maze)+1)
                #df['col'] = np.searchsorted(cols, df['long'])
                #df['row'] = np.searchsorted(rows, df['lat'])
                cols = [abs(i) for i in cols]
                from statistics import mean
                centroid_col = []
                centroid_row = []
                for x, y in zip(cols[0::], cols[1::]):
                    centroid_col.append((x+y)/2)
                for x, y in zip(rows[0::], rows[1::]):
                    centroid_row.append((x+y)/2) 
                #121 get smaller over x going right 
                #37 smaller going down 
                #centroid_col
                #flip order to descend as going down 
                centroid_row = centroid_row[::-1]
                #centroid_row
                import itertools 
                centroid_row_matrix = []
                for i in centroid_row:
                    centroid_row_matrix.append(list(itertools.repeat(i, len(maze.columns))))   
                centroid_col_matrix = []
                for i in centroid_col:
                    centroid_col_matrix.append(list(itertools.repeat(i, len(maze))))
                centroid_col_matrix = pd.DataFrame(centroid_col_matrix).T
                centroid_row_matrix = pd.DataFrame(centroid_row_matrix)
                arr_row = centroid_row_matrix.to_numpy().flatten()
                arr_col = centroid_col_matrix.to_numpy().flatten()
                centroids=zip(arr_row,arr_col)
                centroids = list(centroids)
                #each list is row, starting at top of matrix/plot/image
                gps_map = np.array(centroids)
                params['gps_map'] = gps_map

                np_maze = np.array(maze)
                reduce = skimage.measure.block_reduce(np_maze, (22,22), np.mean)
                maze = pd.DataFrame(reduce)
                maze = maze.astype(int)

                with open(photos_path + '0/coordinates.csv', 'r') as file:
                    data = file.read().replace('\n', ',').strip(',')

                res = data.strip(',').strip("'").split(',')
                res = [float(i) for i in res] 
                bottomLeft = (res[1], res[0])
                bottomRight = (res[7], res[6])
                topLeft = (res[3], res[2])
                topRight = (res[5], res[4])
                cols = np.linspace(bottomLeft[1], bottomRight[1], num=len(maze.columns)+1)
                rows = np.linspace(bottomLeft[0], topLeft[0], num=len(maze)+1)
                #df['col'] = np.searchsorted(cols, df['long'])
                #df['row'] = np.searchsorted(rows, df['lat'])
                cols = [abs(i) for i in cols]
                from statistics import mean
                centroid_col = []
                centroid_row = []
                for x, y in zip(cols[0::], cols[1::]):
                    centroid_col.append((x+y)/2)
                for x, y in zip(rows[0::], rows[1::]):
                    centroid_row.append((x+y)/2) 
                #121 get smaller over x going right 
                #37 smaller going down 
                #centroid_col
                #flip order to descend as going down 
                centroid_row = centroid_row[::-1]
                #centroid_row
                import itertools 
                centroid_row_matrix = []
                for i in centroid_row:
                    centroid_row_matrix.append(list(itertools.repeat(i, len(maze.columns))))   
                centroid_col_matrix = []
                for i in centroid_col:
                    centroid_col_matrix.append(list(itertools.repeat(i, len(maze))))
                centroid_col_matrix = pd.DataFrame(centroid_col_matrix).T
                centroid_row_matrix = pd.DataFrame(centroid_row_matrix)
                arr_row = centroid_row_matrix.to_numpy().flatten()
                arr_col = centroid_col_matrix.to_numpy().flatten()
                centroids=zip(arr_row,arr_col)
                centroids = list(centroids)
                #each list is row, starting at top of matrix/plot/image

                # find smallest distance centroid for drone and fire 
                # find df cell of centroid for drone and fire and assign 
                gps_drone = tuple(params['start_point'])
                #from map point 
                gps_fire = (res[9], res[8])
                def closest_node(node, nodes):
                    nodes = np.asarray(nodes)
                    dist_2 = np.sum((nodes - node)**2, axis=1)
                    return np.argmin(dist_2)
                drone_index = closest_node(gps_drone, centroids)
                fire_index = closest_node(gps_fire, centroids)
                gps_fire = np.array(gps_fire)
                params['end_point'] = gps_fire

                maze_flat = maze.values.flatten().tolist()

                listofzeroes = [0] * len(maze_flat)
                listofzeroes[drone_index] = 'drone'
                listofzeroes[fire_index] = 'fire'
                positions = pd.DataFrame(np.array(listofzeroes).reshape(maze.shape))

                def getIndexes(dfObj, value):
                      
                    # Empty list
                    listOfPos = []
                      
                    # isin() method will return a dataframe with 
                    # boolean values, True at the positions    
                    # where element exists
                    result = dfObj.isin([value])
                      
                    # any() method will return 
                    # a boolean series
                    seriesObj = result.any()
                  
                    # Get list of column names where 
                    # element exists
                    columnNames = list(seriesObj[seriesObj == True].index)
                     
                    # Iterate over the list of columns and
                    # extract the row index where element exists
                    for col in columnNames:
                        rows = list(result[col][result[col] == True].index)
                  
                        for row in rows:
                            listOfPos.append((row, col))
                              
                    # This list contains a list tuples with 
                    # the index of element in the dataframe
                    return listOfPos
                  
                # Calling getIndexes() function to get 
                # the index positions of all occurrences
                # of 22 in the dataframe
                drone = getIndexes(positions, 'drone')
                    
                fire = getIndexes(positions, 'fire')

                drone_coords = str(drone).strip('[]').strip('()').split(', ')
                drone = (int(drone_coords[0]), int(drone_coords[1]))
                fire_coords = str(fire).strip('[]').strip('()').split(', ')
                fire = (int(fire_coords[0]), int(fire_coords[1]))

                maze = maze.to_numpy()
                maze = maze.tolist()

                class Node():
                    """A node class for A* Pathfinding"""
                    def __init__(self, parent=None, position=None):
                        self.parent = parent
                        self.position = position
                        self.g = 0
                        self.h = 0
                        self.f = 0
                    def __eq__(self, other):
                        return self.position == other.position
                def astar(maze, start, end):
                    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
                    # Create start and end node
                    start_node = Node(None, start)
                    start_node.g = start_node.h = start_node.f = 0
                    end_node = Node(None, end)
                    end_node.g = end_node.h = end_node.f = 0
                    # Initialize both open and closed list
                    open_list = []
                    closed_list = []
                    # Add the start node
                    open_list.append(start_node)
                    # Loop until you find the end
                    while len(open_list) > 0:
                        # Get the current node
                        current_node = open_list[0]
                        current_index = 0
                        for index, item in enumerate(open_list):
                            if item.f < current_node.f:
                                current_node = item
                                current_index = index
                        # Pop current off open list, add to closed list
                        open_list.pop(current_index)
                        closed_list.append(current_node)
                        # Found the goal
                        if current_node == end_node:
                            path = []
                            current = current_node
                            while current is not None:
                                path.append(current.position)
                                current = current.parent
                            return path[::-1] # Return reversed path
                        # Generate children
                        children = []
                        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
                            # Get node position
                            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
                            # Make sure within range
                            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                                continue
                            # Make sure walkable terrain
                            if maze[node_position[0]][node_position[1]] != 0:
                                continue
                            # Create new node
                            new_node = Node(current_node, node_position)
                            # Append
                            children.append(new_node)
                        # Loop through children
                        for child in children:
                            # Child is on the closed list
                            for closed_child in closed_list:
                                if child == closed_child:
                                    continue
                            # Create the f, g, and h values
                            child.g = current_node.g + 1
                            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                            child.f = child.g + child.h
                            # Child is already in the open list
                            for open_node in open_list:
                                if child == open_node and child.g > open_node.g:
                                    continue
                            # Add the child to the open list
                            open_list.append(child)
                            
                path = astar(maze, drone, fire)

                z_path = []

                for i in path:
                    z_path.append(list(i))

                for i in z_path:
                    i.append(params['altitude'])

                t_path = []    
                    
                for i in z_path:
                    t_path.append(tuple(i))

                t_path = np.array(t_path)

                params['flight_path_grid'] = t_path 

                fig, ax = plt.subplots()

                plt.pcolor(maze, cmap='plasma', shading='auto', alpha=0.7, edgecolors='black')
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.savefig(photos_path + '0/gridded_density_map.png')


                fig, ax = plt.subplots()

                a,b = map(list,zip(*path))

                ax.scatter(b,a, marker='s', color='black', s=50)

                ax.set_yticks(np.arange(len(positions)))
                ax.set_yticks(np.arange(len(positions)+1)-0.5, minor=True)

                ax.set_xticks(np.arange(len(positions.columns)))
                ax.set_xticks(np.arange(len(positions.columns)+1)-0.5, minor=True)

                ax.grid(True, which="minor")
                ax.set_aspect("equal")


                plt.pcolor(maze, cmap='plasma', shading='auto', alpha=0.7, edgecolor=None)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.savefig(photos_path + '0/flight_path_density_map.png')

                fig, ax = plt.subplots()

                ax.scatter(b,a, marker='s', color='yellow', s=50)

                ax.set_yticks(np.arange(len(positions)))
                ax.set_yticks(np.arange(len(positions)+1)-0.5, minor=True)

                ax.set_xticks(np.arange(len(positions.columns)))
                ax.set_xticks(np.arange(len(positions.columns)+1)-0.5, minor=True)

                ax.grid(True, which="minor")
                ax.set_aspect("equal")

                ax.imshow(img_raw, extent=[0, len(positions.columns), len(positions), 0])
                plt.savefig(photos_path + '0/flight_path_satellite.png')

                maze = pd.DataFrame(maze)

                maze_flat = maze.values.flatten().tolist()

                listofzeroes = [0] * len(maze_flat)

                empty = pd.DataFrame(np.array(listofzeroes).reshape(maze.shape))

                for i in range(len(path)): 
                    empty.iloc[int(str(path[i]).strip('()').split(', ')[0])][int(str(path[i]).strip('()').split(', ')[1])] = i+1

                centroids1 = pd.DataFrame(centroids)
                centroids1 = centroids1.round(4)
                centroidsx = centroids1[0].values.flatten().tolist()
                centroidsy = centroids1[1].values.flatten().tolist()
                centroids1=zip(centroidsx,centroidsy)
                centroids1 = list(centroids1)

                path_flat = empty.values.flatten().tolist()
                zipped = zip(path_flat, centroids1) 
                gps = list(zipped)
                gps = pd.DataFrame(gps)
                gps.columns =['Path_Number', 'Grid_GPS_Centroid']
                df = gps[gps.Path_Number != 0]
                df = df.sort_values(by=['Path_Number'])
                df = df['Grid_GPS_Centroid'].tolist()

                z_path = []

                for i in df:
                    z_path.append(list(i))

                for i in z_path:
                    i.append(int(params['altitude']))

                t_path = []    
                    
                for i in z_path:
                    t_path.append(tuple(i))

                t_path = np.array(t_path)

                params['flight_path'] = t_path 

                pickle.dump(params, open( "params.p", "wb" ))

                

