import os
import numpy as np
import csv
import struct
import json

def remove_coordinates_outside_volume(coordinates,vol_shape):
    """ 
    ---PURPOSE---
    Given an array of x,y,z coordinates and a volume shape,
    remove the coordinates which fall along boundary or outside 
    boundary of volume.

    ---INPUT---
    coordinates  - a 2D array of shape (N,3) with coordinates like:
                   array([x0,y0,z0],...[xN,yN,zN])

    vol_shape    - tuple or list of x,y,z upper bounds like [2160,2560,617]
    ---OUTPUT---
    coordinates[mask] - the subset of coordinates that fall 
                        within the volume
    """
    mask = (coordinates[:,0]<=0) | (coordinates[:,0]>=vol_shape[0]) | \
           (coordinates[:,1]<=0) | (coordinates[:,1]>=vol_shape[1]) | \
           (coordinates[:,2]<=0) | (coordinates[:,2]>=vol_shape[2])
    return coordinates[~mask]

def calculate_factors(level):
    """ 
    ---PURPOSE---
    Calculate the downsampling factor to apply to the grid_shape/chunk size at a given spatial index level.
    This is chosen to make the chunks as isotropic as possible, change as needed for your volume
    ---INPUT---
    level     - 0-indexed integer representing the spatial index level
    ---OUTPUT---
    d[level]  - The downsampling factor to apply to the level to get to the next level
    """
    # 
    d = {}
    d[0] = [1,1,1]
    d[1] = [1,2,2]
    for i in range(2,20):
        d[i] = [2,2,2]
    return d[level]

def make_cells(grid_shape):
    """ 
    ---PURPOSE---
    Make a list of grid cells e.g. ["0_0_0","1_0_0", ...] given a grid shape
    ---INPUT---
    grid_shape  - number of cells at a given level in each coordinate as a list,
                  e.g. [4,4,2] means 4x4x2 grid in x,y,z
    ---OUTPUT---
    cells       - A list of strings representing the cells, 
                  e.g. ['0_0_0', '0_1_0', '1_0_0', '1_1_0']
    """
    cells = []
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            for z in range(grid_shape[2]):
                cell = f"{x}_{y}_{z}"
                cells.append(cell)
    return cells

def get_child_cells(cell,factor):
    """ 
    ---PURPOSE---
    Given a cell string e.g. 1_2_3 and a downsampling factor, e.g. [2,2,1]
    figure out all of the child cells of this cell in the next spatial index level 
    ---INPUT---
    grid_shape  - number of cells at a given level in each coordinate as a list,
                  e.g. [4,4,2] means 4x4x2 grid in x,y,z
    ---OUTPUT---
    cells       - A list of strings representing the cells, 
                  e.g. ['0_0_0', '0_1_0', '1_0_0', '1_1_0']
    """
   
    child_cells = []
    xcell,ycell,zcell = [int(x) for x in cell.split('_')] # n,m,p
    xfactor,yfactor,zfactor = factor # x,y,z
    for xf in range(0,xfactor):
        x_child = xcell*xfactor + xf
        for yf in range(0,yfactor):
            y_child = ycell*yfactor + yf
            for zf in range(0,zfactor):
                z_child = zcell*zfactor + zf
                child_cell = f"{x_child}_{y_child}_{z_child}"
                child_cells.append(child_cell)
    return child_cells

def save_cellfile(level,cell,coordinates,layer_dir,debug=False):
    """ 
    ---PURPOSE---
    Save the binary spatially indexed grid cell file,
    e.g. if level=1 and cell="1_1_0", then the file will be: spatial1/1_1_0 
    Assumes the global variable layer_dir is defined which is the 
    directory in which to save the spatial index directories
    ---INPUT---
    level       - 0-indexed integer representing the spatial index level
    cell        - a string like "0_0_0" representing the x,y,z grid location at a given level 
                  in which you want to extract a subset
    coordinates - a 2D array of coordinates like array([x0,y0,z0],...[xN,yN,zN])
    debug       - if True prints out that it saved the file
    ---OUTPUT---
    Writes the file, but does not return anything
    """
    # We already know how to encode just the coordinates. Do it like so for the first 100 points
    spatial_dir = os.path.join(layer_dir,f"spatial{level}")
    if not os.path.exists(spatial_dir):
        os.mkdir(spatial_dir)
    filename = os.path.join(spatial_dir,cell)
    total_count = len(coordinates)
    with open(filename,'wb') as outfile:
        buf = struct.pack('<Q',total_count)
        pt_buf = b''.join(struct.pack('<3f',x,y,z) for (x,y,z) in coordinates)
        buf += pt_buf
        id_buf = struct.pack('<%sQ' % len(coordinates), *range(len(coordinates)))
        buf += id_buf
        outfile.write(buf)
    if debug:
        print(f"wrote {filename}")
    
def find_intersecting_coordinates(coordinates,lower_bounds,upper_bounds):
    """ 
    ---PURPOSE---
    Find the subset of coordinates that fall within lower and upper bounds in x,y,z
    ---INPUT---
    coordinates  - a 2D array of coordinates like array([x0,y0,z0],...[xN,yN,zN])
    lower_bounds - a tuple or list of x,y,z lower bounds like [0,0,0]
    upper_bounds - a tuple or list of x,y,z upper bounds like [2160,2560,617]
    ---OUTPUT---
    coordinates[mask] - the subset of coordinates that fall 
                        within the lower and upper bounds
    """
    mask = (coordinates[:,0]>=lower_bounds[0]) & (coordinates[:,0]<upper_bounds[0]) & \
           (coordinates[:,1]>=lower_bounds[1]) & (coordinates[:,1]<upper_bounds[1]) & \
           (coordinates[:,2]>=lower_bounds[2]) & (coordinates[:,2]<upper_bounds[2])
    return coordinates[mask]

def make_precomputed_annotation_layer(unique_coordinates,layer_dir,grid_shape = [1,1,1],
         chunk_size=[352,640,540],dimensions_m=[2e-05,2e-05,2e-05],
         limit=10000,debug=False):
    """ 
    ---PURPOSE---
    Create the multiple spatial index levels and save out the cell files at each level.
    Also create, save and return the info file for this layer. All array ordering is x,y,z
    ---INPUT---
    unique_coordinates - An 2D array of shape (N,3) where N is the number of points
                         that you want to spatially index. Rows are objects, columns are x,y,z
                         Duplicates should be removed already.
    layer_dir          - Base precomputed layer directory in which to save the info file
                         and spatial index directories
    grid_shape         - The grid shape of level 0. Typically this is [1,1,1].
    chunk_size         - The chunk size of level 0. If grid_shape = [1,1,1] then this is 
                         the dimensions of the entire volume, e.g. [2160,2560,617]
    dimensions_m       - The x,y,z dimensions in meters in a tuple or list
    limit              - The maximum number of annotations you wish to display 
                         in any cell at any level in Neuroglancer
    debug              - Set to True to print out various quantities to help with debugging
             
    ---OUTPUT---
    Writes out each spatialX/X_Y_Z spatial index file in layer_dir
    Writes out the info file in layer_dir
    info    - a dictionary containing the precomputed info JSON information
    """
    

    # Complete all of the info file except for the spatial part

    info = {}
    info['@type'] = "neuroglancer_annotations_v1"
    info['annotation_type'] = "POINT"
    info['by_id'] = {'key':'by_id'}
    info['dimensions'] = {'x':[str(dimensions_m[0]),'m'],
                          'y':[str(dimensions_m[1]),'m'],
                          'z':[str(dimensions_m[2]),'m']}
    info['lower_bound'] = [0,0,0]
    info['upper_bound'] = chunk_size
    info['properties'] = []
    info['relationships'] = []
    info['spatial'] = []
    
    # Create layer dir if it doesn't exist yet
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)
    
    # initialize some variables
    level=0
    cell="0_0_0"
    
    # Mask to only keep coordinates inside the volume 
    # coordinates along outside boundary of volume will be discarded
    unique_coordinates = remove_coordinates_outside_volume(
        unique_coordinates,chunk_size)

    # Shuffle coordinates
    np.random.shuffle(unique_coordinates)

    total_annotations = len(unique_coordinates)
    remaining_annotations = {} # will hold the arrays of coordinates in each cell at each level
    remaining_annotations[level] = {cell:unique_coordinates}

    maxCount = {} # will hold the maximum remaining annotations at each level
    
    # Iterate over levels until there are no more annotations to assign to child cells
    while True:
        if debug:
            print("##############")
            print(f"Level: {level}")
            print("##############")
        
        # Figure out maxCount to see if we are out of cells
        N_annotations_this_level = [len(x) for x in remaining_annotations[level].values()]
        maxCount[level] = max(N_annotations_this_level)
        if maxCount[level] == 0:
            print("Finished! Writing info file:")
            info_path = os.path.join(layer_dir,"info")
            print(info_path)
            with open(info_path,'w') as outfile:
                json.dump(info,outfile,indent=2)
            break
        # If we made it past there then we have cells left to assign
    
        # Use utility functions to figure out grid_shape and chunk_size for this level
        factor = calculate_factors(level)
        grid_shape = [a*b for a,b in zip(grid_shape,factor)]
        chunk_size = [a/b for a,b in zip(chunk_size,factor)]
        # Make the spatial dict for the info file
        spatial_dict_this_level = {
        'key':f'spatial{level}',
        'grid_shape':grid_shape,
        'chunk_size':chunk_size,
        'limit':limit
        }
        info['spatial'].append(spatial_dict_this_level)
        
        cells = make_cells(grid_shape)
            
        if debug:
            print(f"chunk_size={chunk_size}, maxCount = {maxCount[level]}")
            print("Have these cells:", cells)
        
        # Figure out the probability of extracting each annotation based on the limit
        if maxCount[level] > limit:
            prob = limit/maxCount[level]
        else:
            prob = 1
            
        # Loop over each cell at this level
        for cell in cells:
            if debug:
                print("In cell: ", cell)
            
            # Look up the remaining annotations in this cell, which was computed during the last iteration
            annotations_this_cell = remaining_annotations[level][cell]            
            N_annotations_this_cell = len(annotations_this_cell)
            if debug:
                print(f"started with {N_annotations_this_cell} annotations")
            
            # Need to know the child cells and the size of each so we can figure out the 
            # remaining counts in each
            next_factor = calculate_factors(level+1)
            child_cells = get_child_cells(cell,next_factor)
            next_chunk_size = [a/b for a,b in zip(chunk_size,next_factor)]

            # If we have annotations in this cell, then save the spatial index file for this level and cell
            # If not, don't save the file since it would be empty
            if N_annotations_this_cell != 0:
                # Figure out the subset of cells based on the probability calculated above
                N_subset = int(round(N_annotations_this_cell*prob))
                
                # figure out list of indices of the remaining array to grab 
                subset_indices = np.random.choice(range(N_annotations_this_cell),size=N_subset,replace=False)
                # Use these indices to get the subset of annotations
                subset_cells = np.take(annotations_this_cell,subset_indices,axis=0)
                
                if debug:
                    print(f"subsetted {len(subset_cells)} annotations")
                # save these cells to a spatial index file
                save_cellfile(level,cell,subset_cells,layer_dir,debug=debug)
                
                # Figure out the leftover annotations that weren't included in the subset
                indices_annotations_this_cell = range(len(annotations_this_cell))
                leftover_annotation_indices = set(indices_annotations_this_cell)-set(subset_indices)
                leftover_annotations = np.take(annotations_this_cell,list(leftover_annotation_indices),axis=0)
                if debug:
                    print(f"have {len(leftover_annotations)} annotations leftover")
            else:
                leftover_annotations = np.array([])
            # Initialize the next level in the remaining_annotations dictionary
            if level+1 not in remaining_annotations.keys():
                remaining_annotations[level+1] = {}
            
            if debug:
                print("Looping over child cells: ", child_cells)
            
            # Intiailize a variable to keep track of how many annotations total are in each child cell
            n_annotations_in_child_cells = 0
            
            # Loop over child cells and figure out how many of the remaining annotations 
            # fall in each child cell region
            for child_cell in child_cells:
                if N_annotations_this_cell == 0:
                    remaining_annotations[level+1][child_cell] = np.array([])
                    continue
                
                if debug:
                    print(f"Child cell: {child_cell}")
                
                # figure out which of the leftover annotations fall within this child cell
                child_cell_indices = [int(x) for x in child_cell.split('_')]
                child_lower_bounds = [a*b for a,b in zip(child_cell_indices,next_chunk_size)]
                child_upper_bounds = [a+b for a,b, in zip(child_lower_bounds,next_chunk_size)]
                
                if debug:
                    print("Child lower and upper bounds")
                    print(child_lower_bounds)
                    print(child_upper_bounds)

                # Now use the bounds to find intersecting annotations in this child cell
                intersecting_annotations_this_child = find_intersecting_coordinates(
                    leftover_annotations,child_lower_bounds,child_upper_bounds)
                
                if debug:
                    print(f"Have {len(intersecting_annotations_this_child)} in this child cell")
                
                # Assign the remaining annotations for the child cell in the dictionary
                remaining_annotations[level+1][child_cell] = intersecting_annotations_this_child
                
                n_annotations_in_child_cells+=len(intersecting_annotations_this_child)
            
            # Make sure that the sum of all annotations in all child cells equals the total for this cell
            if debug:
                print("Leftover annotations this cell vs. sum in child cells")
                print(len(leftover_annotations),n_annotations_in_child_cells)
        assert len(leftover_annotations) == n_annotations_in_child_cells, "This is likely due to having annotations along the edge of your volume. Remove these annotations and try re-running. "
        
        # increment to the next level before next iteration in while loop
        level+=1
    return info