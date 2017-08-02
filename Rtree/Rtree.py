import time

# build a node class
class Node(object):
    def __init__(self, mbr, points, nop, index, left = None, right = None):
        self.mbr = mbr
        self.points = points
        self.nop = nop
        self.index = index
        self.left = left
        self.right = right
    def isLeafNode(self):
        if self.nop <= 16:
            return True
        else:
            return False
    def __repr__(self):
        return self.mbr
    def __str__(self):
        return str(self.mbr)


# read all data from a txt file
def read_dataset(filename):
    readfile = open(filename, 'r')  # 'r' means read
    datalist = []
    for line in readfile:
        datalist.append(line.split())
    return datalist

# get the rectangle of all data points
def get_data_rectangle(datalist):
    datalist.sort(key = lambda x:int(x[1]))
    xmin, xmax = datalist[0][1], datalist[len(datalist)-1][1]
    datalist.sort(key = lambda x:int(x[2]))
    ymin, ymax = datalist[0][2], datalist[len(datalist)-1][2]
    return [[xmin, ymin], [xmax, ymax]]

# sort data by the latitude value of data
# initially divide data as (40%, 60%), then (40%+1, 60%-1)... until (60%, 40%)
# get minimum perimeter of leftside mbr plus rightside mbr
def get_optimal_mbr_by_x(datalist):
    datalist_size = len(datalist)
    datalist.sort(key = lambda x:int(x[1]))  # sort data by latitude
    leftside_data = datalist[:int(datalist_size * 0.4)]
    rightside_data = datalist[int(datalist_size * 0.4):]
    left_mbr = [[int(leftside_data[0][1]), min([int(i[2]) for i in leftside_data])], 
                [int(leftside_data[len(leftside_data)-1][1]), max([int(i[2]) for i in leftside_data])]]
    left_mbr_perimeter = ((left_mbr[1][0] - left_mbr[0][0]) + (left_mbr[1][1] - left_mbr[0][1])) * 2
    right_mbr = [[int(rightside_data[0][1]), min([int(i[2]) for i in rightside_data])], 
                [int(rightside_data[len(rightside_data)-1][1]), max([int(i[2]) for i in rightside_data])]]
    right_mbr_perimeter = ((right_mbr[1][0] - right_mbr[0][0]) + (right_mbr[1][1] - right_mbr[0][1])) * 2
    perimeters = {int(datalist_size * 0.4 - 1) : left_mbr_perimeter + right_mbr_perimeter}  # create dictionary for perimeters
    for i in range(int(datalist_size * 0.4), int(datalist_size * 0.6)):
        left_mbr_perimeter += (int(datalist[i][1]) - left_mbr[1][0]) * 2  # width increased
        left_mbr[1][0] = int(datalist[i][1])  # set new max latitude in leftside
        if int(datalist[i][2]) > left_mbr[1][1]:  # new longitude is greater than max longitude
            left_mbr_perimeter += (int(datalist[i][2]) - left_mbr[1][1]) * 2  # height increased
            left_mbr[1][1] = int(datalist[i][2])  # set new max longitude in leftside
        elif int(datalist[i][2]) < left_mbr[0][1]:  # new longitude is smaller than min longitude
            left_mbr_perimeter += (left_mbr[0][1] - int(datalist[i][2])) * 2  # height increased
            left_mbr[0][1] = int(datalist[i][2])  # set new min longitude in leftside
        right_mbr_perimeter -= (int(datalist[i+1][1]) - int(datalist[i][1])) * 2  # width reduced
        right_mbr[1][0] = int(datalist[i+1][1])  # set new max latitude in rightside
        if int(datalist[i][2]) == right_mbr[1][1]:  # chosen longitude is current max longitude
            # get max longitude among rightside datalist without datalist[i],
            # and set it as new max longitude in rightside
            right_mbr[1][1] = max([int(i[2]) for i in datalist[(i+1):]])
            right_mbr_perimeter -= (int(datalist[i][2]) - right_mbr[1][1]) * 2  # height reduced
        elif int(datalist[i][2]) == right_mbr[0][1]:  # chosen longitude is current min longitude
            # get min longitude among rightside datalist without datalist[i],
            # and set it as new min longitude in rightside
            right_mbr[0][1] = min([int(i[2]) for i in datalist[(i+1):]])
            right_mbr_perimeter -= (right_mbr[0][1] - int(datalist[i][2])) * 2  # height reduced
        perimeters[i] = left_mbr_perimeter + right_mbr_perimeter  # add new i:perimeter into perimeters
    #return min(perimeters, key = perimeters.get)  # get the index of the minumum perimeter
    return perimeters

# sort data by the longitude value of data
# initially divide data as (40%, 60%), then (40%+1, 60%-1)... until (60%, 40%)
# get minimum perimeter of topside mbr plus bottomside mbr
def get_optimal_mbr_by_y(datalist):
    datalist_size = len(datalist)
    datalist.sort(key = lambda x:int(x[2]))  # sort data by longitude
    bottomside_data = datalist[:int(datalist_size * 0.4)]
    topside_data = datalist[int(datalist_size * 0.4):]
    bottom_mbr = [[min([int(i[1]) for i in bottomside_data]), int(bottomside_data[0][2])], 
                 [max([int(i[1]) for i in bottomside_data]), int(bottomside_data[len(bottomside_data)-1][2])]]
    bottom_mbr_perimeter = ((bottom_mbr[1][0] - bottom_mbr[0][0]) + (bottom_mbr[1][1] - bottom_mbr[0][1])) * 2
    top_mbr = [[min([int(i[1]) for i in topside_data]), int(topside_data[0][2])], 
                 [max([int(i[1]) for i in topside_data]), int(topside_data[len(topside_data)-1][2])]]
    top_mbr_perimeter = ((top_mbr[1][0] - top_mbr[0][0]) + (top_mbr[1][1] - top_mbr[0][1])) * 2
    perimeters = {int(datalist_size * 0.4) - 1 : bottom_mbr_perimeter + top_mbr_perimeter}  # create dictionary for perimeters
    for i in range(int(datalist_size * 0.4), int(datalist_size * 0.6)):
        bottom_mbr_perimeter += (int(datalist[i][2]) - bottom_mbr[1][1]) * 2  # height increased
        bottom_mbr[1][1] = int(datalist[i][2])  # set new max longitude in bottomside
        if int(datalist[i][1]) > bottom_mbr[1][0]:  # new latitude is greater than max latitude
            bottom_mbr_perimeter += (int(datalist[i][1]) - bottom_mbr[1][0]) * 2  # width increased
            bottom_mbr[1][0] = int(datalist[i][1])  # set new max latitude in bottomside
        elif int(datalist[i][1]) < bottom_mbr[0][0]:  # new latitude is smaller than min latitude
            bottom_mbr_perimeter += (bottom_mbr[0][0] - int(datalist[i][1])) * 2  # width increased
            bottom_mbr[0][0] = int(datalist[i][1])  # set new min latitude in bottomside
        top_mbr_perimeter -= (int(datalist[i+1][2]) - int(datalist[i][2])) * 2  # height reduced
        top_mbr[1][1] = int(datalist[i+1][2])  # set new max longitude in topside
        if int(datalist[i][1]) == top_mbr[1][0]:  # chosen latitude is current max latitude
            # get max latitude among topside datalist without datalist[i],
            # and set it as new max latitude in topside
            top_mbr[1][0] = max([int(i[1]) for i in datalist[(i+1):]])
            top_mbr_perimeter -= (int(datalist[i][1]) - top_mbr[1][0]) * 2  # width reduced
        elif int(datalist[i][1]) == top_mbr[0][0]:  # chosen latitude is current min latitude
            # get min latitude among topside datalist without datalist[i],
            # and set it as new min latitude in topside
            top_mbr[0][0] = min([int(i[1]) for i in datalist[(i+1):]])
            top_mbr_perimeter -= (top_mbr[0][0] - int(datalist[i][1])) * 2  # width reduced
        perimeters[i] = bottom_mbr_perimeter + top_mbr_perimeter  # add new i:perimeter into perimeters
    #print (min(perimeters, key = perimeters.get))  # get the index of the minumum perimeter
    return perimeters

# compare data horizontal and vertical distribution results
# return the number of points within divided two blocks respectively
def get_number_of_points(datalist):
    x = min(get_optimal_mbr_by_x(datalist).values())
    y = min(get_optimal_mbr_by_y(datalist).values())
    x_number_of_points = min(get_optimal_mbr_by_x(datalist), key = get_optimal_mbr_by_x(datalist).get) + 1
    y_number_of_points = min(get_optimal_mbr_by_y(datalist), key = get_optimal_mbr_by_y(datalist).get) + 1
    if x >= y:  # vertical distribution is better
        return [y_number_of_points, len(datalist) - y_number_of_points, 'y']
    else:  # horizontal distribution is better
        return [x_number_of_points, len(datalist) - x_number_of_points, 'x']

# get all points within divided two blocks respectively
def get_points(datalist):
    num = get_number_of_points(datalist)
    if num[2] == 'y':
        datalist.sort(key = lambda x:int(x[2]))  # sort data by longitude
    else:
        datalist.sort(key = lambda x:int(x[1]))  # sort data by latitude
    return [datalist[:num[0]], datalist[num[0]:]]

# return coordinates of two divided optimal MBRs respectively
def get_optimal_mbr(datalist):
    num = get_number_of_points(datalist)
    mbr = []
    if num[2] == 'y':  # vertical distribution is better
        datalist.sort(key = lambda x:int(x[2]))  # sort data by longitude
        mbr.append([[min([int(i[1]) for i in datalist[:num[0]]]), int(datalist[0][2])],
                   [max([int(i[1]) for i in datalist[:num[0]]]), int(datalist[num[0]-1][2])]])
        mbr.append([[min([int(i[1]) for i in datalist[num[0]:]]), int(datalist[num[0]][2])],
                   [max([int(i[1]) for i in datalist[num[0]:]]), int(datalist[len(datalist)-1][2])]])
    else:  # horizontal distribution is better
        datalist.sort(key = lambda x:int(x[1]))  # sort data by latitude
        mbr.append([[int(datalist[0][1]), min([int(i[2]) for i in datalist[:num[0]]])],
                    [int(datalist[num[0]-1][1]), max([int(i[2]) for i in datalist[:num[0]]])]])
        mbr.append([[int(datalist[num[0]][1]), min([int(i[2]) for i in datalist[num[0]:]])],
                    [int(datalist[len(datalist)-1][1]), max([int(i[2]) for i in datalist[num[0]:]])]])
    mbr_perimeter = ((int(mbr[0][1][0]) - int(mbr[0][0][0])) + (int(mbr[0][1][1]) - int(mbr[0][0][1])) 
        + (int(mbr[1][1][0]) - int(mbr[1][0][0])) + (int(mbr[1][1][1]) - int(mbr[1][0][1]))) * 2
    return mbr

# build the tree
def get_tree(datalist, l_tree, r_tree):
    if l_tree.isLeafNode() and not r_tree.isLeafNode():
        rl_tree_mbr = get_optimal_mbr(datalist[1])[0]
        rr_tree_mbr = get_optimal_mbr(datalist[1])[1]
        rl_tree_points = get_points(datalist[1])[0]
        rr_tree_points = get_points(datalist[1])[1]
        rl_tree_nop = len(rl_tree_points)
        rr_tree_nop = len(rr_tree_points)
        rl_index = [[r_tree.index[0][0] + 1], [r_tree.index[1][0] * 2]]
        rr_index = [[r_tree.index[0][0] + 1], [r_tree.index[1][0] * 2 + 1]]
        rl_tree = Node(rl_tree_mbr, rl_tree_points, rl_tree_nop, rl_index)
        rr_tree = Node(rr_tree_mbr, rr_tree_points, rr_tree_nop, rr_index)
        r_tree.left = rl_tree
        r_tree.right = rr_tree
        get_tree(get_points(datalist[1]), rl_tree, rr_tree)
    elif not l_tree.isLeafNode() and r_tree.isLeafNode():
        ll_tree_mbr = get_optimal_mbr(datalist[0])[0]
        lr_tree_mbr = get_optimal_mbr(datalist[0])[1]
        ll_tree_points = get_points(datalist[0])[0]
        lr_tree_points = get_points(datalist[0])[1]
        ll_tree_nop = len(ll_tree_points)
        lr_tree_nop = len(lr_tree_points)
        ll_index = [[l_tree.index[0][0] + 1], [l_tree.index[1][0] * 2]]
        lr_index = [[l_tree.index[0][0] + 1], [l_tree.index[1][0] * 2 + 1]]
        ll_tree = Node(ll_tree_mbr, ll_tree_points, ll_tree_nop, ll_index)
        lr_tree = Node(lr_tree_mbr, lr_tree_points, lr_tree_nop, lr_index)
        l_tree.left = ll_tree
        l_tree.right = lr_tree
        get_tree(get_points(datalist[0]), ll_tree, lr_tree)
    elif not l_tree.isLeafNode() and not r_tree.isLeafNode():
        ll_tree_mbr = get_optimal_mbr(datalist[0])[0]
        lr_tree_mbr = get_optimal_mbr(datalist[0])[1]
        ll_tree_points = get_points(datalist[0])[0]
        lr_tree_points = get_points(datalist[0])[1]
        ll_tree_nop = len(ll_tree_points)
        lr_tree_nop = len(lr_tree_points)
        ll_index = [[l_tree.index[0][0] + 1], [l_tree.index[1][0] * 2]]
        lr_index = [[l_tree.index[0][0] + 1], [l_tree.index[1][0] * 2 + 1]]
        ll_tree = Node(ll_tree_mbr, ll_tree_points, ll_tree_nop, ll_index)
        lr_tree = Node(lr_tree_mbr, lr_tree_points, lr_tree_nop, lr_index)
        l_tree.left = ll_tree
        l_tree.right = lr_tree
        get_tree(get_points(datalist[0]), ll_tree, lr_tree)
        rl_tree_mbr = get_optimal_mbr(datalist[1])[0]
        rr_tree_mbr = get_optimal_mbr(datalist[1])[1]
        rl_tree_points = get_points(datalist[1])[0]
        rr_tree_points = get_points(datalist[1])[1]
        rl_tree_nop = len(rl_tree_points)
        rr_tree_nop = len(rr_tree_points)
        rl_index = [[r_tree.index[0][0] + 1], [r_tree.index[1][0] * 2]]
        rr_index = [[r_tree.index[0][0] + 1], [r_tree.index[1][0] * 2 + 1]]
        rl_tree = Node(rl_tree_mbr, rl_tree_points, rl_tree_nop, rl_index)
        rr_tree = Node(rr_tree_mbr, rr_tree_points, rr_tree_nop, rr_index)
        r_tree.left = rl_tree
        r_tree.right = rr_tree
        get_tree(get_points(datalist[1]), rl_tree, rr_tree)

# return rectangles as formatted [x_min, x_max, y_min, y_max]
def get_query_rectangles(range_queries):
    rectangles = []
    for range_query in range_queries:       
        rectangles.append([min(int(range_query[0]), int(range_query[1])), max(int(range_query[0]), int(range_query[1])),
                           min(int(range_query[2]), int(range_query[3])), max(int(range_query[2]), int(range_query[3]))])  
    return rectangles

# check if two rectangles are intersecting
def check_intersection(rec, mbr):
    # not intersecting
    if rec[0] > mbr[1][0] or rec[1] < mbr[0][0] or rec[2] > mbr[1][1] or rec[3] < mbr[0][1]:
        return 0
    # intersecting and covered
    elif rec[1] >= mbr[1][0] and rec[3] >= mbr[1][1] and rec[0] <= mbr[0][0] and rec[2] <= mbr[0][1]:
        return 1

# check if a point is inside a rectangle
def check_is_inside(rec, point):
    if int(point[1]) > rec[0] and int(point[1]) < rec[1] and int(point[2]) > rec[2] and int(point[2]) < rec[3]:
        return True
    else:
        return False

# test range query
def range_query_testing(rec, tree, count):
    intersect_result = check_intersection(rec, tree.mbr)  
    if intersect_result == 1:
        count += tree.nop
    elif intersect_result != 0:
        if tree.isLeafNode():
            for point in tree.points:
                if check_is_inside(rec, point):
                    count += 1
        else:
            return range_query_testing(rec, tree.left, count) + range_query_testing(rec, tree.right, count)
    return count

# return the minimum distance between point to mbr
def point_to_mbr_distance(nn, mbr):
    # inside
    if int(nn[0]) >= mbr[0][0] and int(nn[0]) <= mbr[1][0] and int(nn[1]) >= mbr[0][1] and int(nn[1]) <= mbr[1][1]:
        return 0
    # left or right side
    elif int(nn[1]) > mbr[0][1] and int(nn[1]) < mbr[1][1]:
        if int(nn[0]) < mbr[0][0]: # left side
            return mbr[0][0] - int(nn[0])
        elif int(nn[0]) > mbr[1][0]: # right side
            return int(nn[0]) - mbr[1][0]
    # top or bottom side
    elif int(nn[0]) > mbr[0][0] and int(nn[0]) < mbr[1][0]:
        if int(nn[1]) < mbr[0][1]: # bottom side
            return mbr[0][1] - int(nn[1])
        elif int(nn[1]) > mbr[1][1]: # top side
            return int(nn[1]) - mbr[1][1]
    # top right corner
    elif int(nn[0]) > mbr[1][0] and int(nn[1]) > mbr[1][1]:
        return ((int(nn[0]) - mbr[1][0])**2 + (int(nn[1]) - mbr[1][1])**2)**(1/2)
    # top left corner
    elif int(nn[0]) < mbr[0][0] and int(nn[1]) > mbr[1][1]:
        return ((mbr[0][0] - int(nn[0]))**2 + (int(nn[1]) - mbr[1][1])**2)**(1/2)
    # bottom left corner
    elif int(nn[0]) < mbr[0][0] and int(nn[1]) < mbr[0][1]:
        return ((mbr[0][0] - int(nn[0]))**2 + (mbr[0][1] - int(nn[1]))**2)**(1/2)
    # bottom right corner
    elif int(nn[0]) > mbr[1][0] and int(nn[1]) < mbr[0][1]:
        return ((int(nn[0]) - mbr[1][0])**2 + (mbr[0][1] - int(nn[1]))**2)**(1/2)


def nn_query_testing(nn, tree, queue):
    if tree.isLeafNode():
        _id = []
        distances = []
        for point in tree.points:
            distances.append((int(point[1])-int(nn[0]))**2+(int(point[2])-int(nn[1]))**2)
        min_value = min(distances)
        i = 0
        # if there are more than one minimum value
        for distance in distances:
            if distance == min_value:
                _id.append(tree.points[i][0])
            i += 1
        return _id
    else:
        l_distance = point_to_mbr_distance(nn, tree.left.mbr)
        r_distance = point_to_mbr_distance(nn, tree.right.mbr)
        queue.append([tree.left, l_distance])
        queue.append([tree.right, r_distance])
        for q in queue:
            if not isinstance(q[1], int):
                queue.remove(q)
        queue.sort(key = lambda x:x[1])  # sort by distance from low to high
        tree = queue[0][0]
        queue.pop(0)  # get rid of the nearest non-leaf-node mbr
        return nn_query_testing(nn, tree, queue)


# the brutal force way to test range queries that doe not use r tree
def original_range_query_testing(recs, datalist):
    counts = []
    for rec in recs:
        count = 0
        for point in datalist:
            if int(point[1]) >= rec[0] and int(point[1]) <= rec[1] and int(point[2]) >= rec[2] and int(point[2]) <= rec[3]:
                count += 1
        counts.append(count)
    return counts

# the brutal force way to test nearest neighbor queries
def original_nn_query_testing(nns, datalist):
    ids = []
    for nn in nns:
        distances = []
        _id = []
        for point in datalist:
            distances.append((int(point[1])-int(nn[0]))**2+(int(point[2])-int(nn[1]))**2)
        min_value = min(distances)
        i = 0
        # if there are more than one minimum value
        for distance in distances:
            if distance == min_value:
                _id.append(datalist[i][0])
            i += 1
        ids.append(_id)
    return ids

#sequential scan   
start_time = time.time()
datalist = read_dataset("F:/dataSet100k.txt")
datalist.pop(0)
elapsed_time = time.time() - start_time  # get time cost
print(elapsed_time)  # whole time

range_queries = read_dataset("F:/rangeDataSet.txt")
rectangles = get_query_rectangles(range_queries)
nns = read_dataset("F:/nearestNeighborDataSet.txt")

# brutal force result and time testing of range query
start_time_1 = time.time()
print (original_range_query_testing(rectangles, datalist))
brutal_force_range = original_range_query_testing(rectangles, datalist)
elapsed_time_1 = time.time() - start_time_1  # get time cost
print(elapsed_time_1)  # whole time
print(elapsed_time_1 / 100)  # average time

# brutal force result and time testing of nearest neighbor query
start_time_2 = time.time()
print (original_nn_query_testing(nns, datalist))
brutal_force_nn = original_nn_query_testing(nns, datalist)
elapsed_time_2 = time.time() - start_time_2  # get time cost
print(elapsed_time_2)  # whole time
print(elapsed_time_2 / 100)  # average time

# build tree root
tree = Node([], datalist, len(datalist), [[0],[0]])
l_tree_mbr = get_optimal_mbr(datalist)[0]
r_tree_mbr = get_optimal_mbr(datalist)[1]
datalist = get_points(datalist)
l_tree_points = datalist[0]
r_tree_points = datalist[1]
l_tree_nop = len(l_tree_points)
r_tree_nop = len(r_tree_points)
l_tree_index = [[1],[0]]
r_tree_index = [[1],[1]]
l_tree = Node(l_tree_mbr, l_tree_points, l_tree_nop, l_tree_index)
r_tree = Node(r_tree_mbr, r_tree_points, r_tree_nop, r_tree_index)
tree.left = l_tree
tree.right = r_tree

# build the whole tree
get_tree(datalist, l_tree, r_tree)

# range query result and time testing with r tree 
start_time_3 = time.time()
points = []
for rectangle in rectangles:
    points.append(range_query_testing(rectangle, tree.left, 0) + range_query_testing(rectangle, tree.right, 0))
print(points)
elapsed_time_3 = time.time() - start_time_3  # get time cost
print(elapsed_time_3)  # whole time
print(elapsed_time_3 / 100)  # average time


# nearest neighbor query result and time testing with r tree 
start_time_4 = time.time()
ids = []
for nn in nns:
    ids.append(nn_query_testing(nn, tree, []))
print(ids_)
elapsed_time_4 = time.time() - start_time_4  # get time cost
print(elapsed_time_4)  # whole time
print(elapsed_time_4 / 100)  # average time

f = open('result.txt', 'w+')
f.write('Sequential Scan Time: ' + str(elapsed_time) + '\n')
f.write('\n')
f.write('Range Query Test Brutal Force Result: \n')
f.write('Total time: ' + str(elapsed_time_1) + '\n')
f.write('Average time: ' + str(elapsed_time_1 / 100) + '\n')
f.write('NN Query Test Brutal Force Result: \n')
f.write('Total time: ' + str(elapsed_time_2) + '\n')
f.write('Average time: ' + str(elapsed_time_2 / 100) + '\n')
f.write('Range Query Test With R-Tree Result: \n')
for i in points:
  f.write("%s" % [i])
f.write('\n')
f.write('Total time: ' + str(elapsed_time_3) + '\n')
f.write('Average time: ' + str(elapsed_time_3 / 100) + '\n')
f.write('Efficiency: ' + str(elapsed_time_1 / elapsed_time_3) + 'times faster \n')
f.write('\n')
f.write('NN Query Test With R-Tree Result: \n')
for i in ids:
  f.write("%s" % i)
f.write('\n')
f.write('Total time: ' + str(elapsed_time_4) + '\n')
f.write('Average time: ' + str(elapsed_time_4 / 100) + '\n')
f.write('Efficiency: ' + str(elapsed_time_2 / elapsed_time_4) + 'times faster \n')
f.close()
