# This file has some helper functions to work with mtri and gmsh geometries.
# These are used to debug readOH2csg tool: omegah2degas2


import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import matplotlib.tri as mtri
import pickle
import gmsh

def get_neighboring_cells(surface_id, surfaces, neighbors):
    neg_pointer = surfaces[surface_id][0,0] # adjust for 0-based indexing
    pos_pointer = surfaces[surface_id][0,1]  # adjust for 0-based indexing
    neg_count = surfaces[surface_id][1,0]
    pos_count = surfaces[surface_id][1,1]
    #print(f"Surface {surface_id}: neg={neg_pointer}, pos={pos_pointer}, neg_count={neg_count}, pos_count={pos_count}")
    
    neg_cells = neighbors[neg_pointer:neg_pointer+neg_count]
    pos_cells = neighbors[pos_pointer:pos_pointer+pos_count]
    #print(f"  Neighboring cells on negative side: {neg_cells}")
    #print(f"  Neighboring cells on positive side: {pos_cells}")
    return neg_cells, pos_cells

def parse_definegeometry2d_terminal_output(file_path):
    """
    Parse a triangulation file and extract all triangulation blocks.
    
    Parameters:
    -----------
    file_path : str
        Path to the triangulation file
        
    Returns:
    --------
    list of dict
        Each dict contains:
        - 'points': list of (x, y) tuples
        - 'triangles': list of triangles, each as a list of point indices
        - 'segments': list of segments, each as a list of two point indices
    """
    triangulations = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for the start of a triangulation block
        if line.startswith("Final triangulation:"):
            points = []
            triangles = []
            
            i += 1  # Move past the header
            
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Parse points
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Point"):
                    parts = line.split()
                    # Format: "Point    0:  1.5  -0.5"
                    x = float(parts[2])
                    y = float(parts[3])
                    points.append((x, y))
                    i += 1
                else:
                    break
            
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Parse triangles
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Triangle"):
                    parts = line.split()
                    # Format: "Triangle    0 points:     2     1     0"
                    idx1 = int(parts[3])
                    idx2 = int(parts[4])
                    idx3 = int(parts[5])
                    triangles.append([idx1, idx2, idx3])
                    i += 1
                else:
                    break
            
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Parse segments
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Segment"):
                    parts = line.split()
                    # Format: "Segment    0 points:     0     1"
                    idx1 = int(parts[3])
                    idx2 = int(parts[4])
                    i += 1
                else:
                    break
            
            # Store this triangulation
            if points and triangles:
                triangulations.append({
                    'points': points,
                    'triangles': triangles,
                })
        else:
            i += 1
    
    return triangulations

def plot_triangulation_with_labels(triang, title='Mesh', xlim=None, ylim=None, axis=None):
    """
    Plot a triangulation with node and triangle labels.
    
    Parameters:
    -----------
    triang : matplotlib.tri.Triangulation
        The triangulation object to plot
    title : str, optional
        Title for the plot (default: 'Mesh')
    xlim : tuple, optional
        X-axis limits as (xmin, xmax)
    ylim : tuple, optional
        Y-axis limits as (ymin, ymax)
    """
    if axis is None:
        fig, axis = plt.subplots(figsize=(10, 8))
    
    # Plot the triangulation
    axis.triplot(triang, color='blue')
    axis.plot(triang.x, triang.y, 'o', color='red')
    
    # Show node numbers
    for i, (x, y) in enumerate(zip(triang.x, triang.y)):
        axis.text(x, y, str(i), color='black', fontsize=12, ha='right', va='bottom')
    
    # Show triangle numbers
    for i, triangle in enumerate(triang.triangles):
        x = np.mean(triang.x[triangle])
        y = np.mean(triang.y[triangle])
        axis.text(x, y, str(i), color='green', fontsize=12, ha='center', va='center')
    
    axis.set_xlabel('X-axis')
    axis.set_ylabel('Y-axis')
    axis.set_title(title)
    
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)
    
    axis.set_aspect('equal')

def get_cell_boundaries(cell_id, cells, boundaries):
    """
    Get all boundary IDs (0-indexed) for a given cell ID and sign indicating orientation.
    Sign indicates orientation (positive for outward, negative for inward).
    
    Parameters:
        cell_id (int): ID of the cell (0-indexed)
        cells (list): List of cell definitions, where each cell is defined as 
                      (start_boundary, n_boundary, n_tot_boundary, zone_id) like in degas2 `geometry.nc`
        boundaries (list): List of boundary IDs (signed for orientation) - 1-indexed like in degas2 `geometry.nc`
    Returns:
        list: List of boundary IDs (0-indexed) for the specified cell and their signs
    """
    start_boundary, n_boundary, n_tot_boundary, zone_id = cells[cell_id]
    cell_boundary_ids = boundaries[start_boundary-1:start_boundary+n_tot_boundary-1]
    
    signs = np.sign(cell_boundary_ids)
    assert np.all(signs != 0), "Boundary IDs should not be zero."
        
    # make boundary IDs 0-indexed: +1 for negative IDs, -1 for positive IDs
    cell_boundary_ids = np.abs(cell_boundary_ids) - 1
    
    
    return cell_boundary_ids, signs

#assert np.array_equal(get_cell_boundaries(0), (np.array([0,1,2,3]), np.array([1,1,1,1]))), f"Boundary IDs for cell 0 do not match expected values: {get_cell_boundaries(0)}"
#assert np.array_equal(get_cell_boundaries(1), (np.array([6,7, 5]), np.array([1,-1,1]))), f"Boundary IDs for cell 1 do not match expected values: {get_cell_boundaries(1)}"


def evaluate_dg2_quadric(coeffs, point):
    """
    Evaluate the quadric surface equation at a given point.
    
    Parameters:
        coeffs (array-like): Coefficients [c0, cx, cy, cz, cxx, cyy, czz, cxy, cyz, cxz]
        point (array-like): Point to evaluate, shape (3,) representing (x, y, z)
        
    Returns:
        float: Evaluated value at the given point
    """
    c0, cx, cy, cz, cxx, cyy, czz, cxy, cyz, cxz = coeffs
    x = point[0]
    y = point[1]
    z = point[2]
    
    value = (c0 + cx * x + cy * y + cz * z +
              cxx * x**2 + cyy * y**2 + czz * z**2 +
              cxy * x * y + cyz * y * z + cxz * x * z)
    return value

def plot_quad_on_xz(coeffs, x_range, z_range, num_points=400, axis=None, sign=None):
    """
    Plot the quadric surface projected onto the XZ plane which becomes a line in XZ.
    
    Parameters:
        coeffs (array-like): Coefficients [c0, cx, cy, cz, cxx, cyy, czz, cxy, cyz, cxz]
        x_range (tuple): Range of x values (xmin, xmax)
        z_range (tuple): Range of z values (zmin, zmax)
        num_points (int): Number of points in each dimension for the grid
        axis: Matplotlib axis to plot on (if None, uses plt)
        sign (int, optional): If +1, fill positive region; if -1, fill negative region with red (opacity=0.1)
    """
    
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    z_vals = np.linspace(z_range[0], z_range[1], num_points)
    
    # plot the implicit contour where the quadric evaluates to zero
    X, Z = np.meshgrid(x_vals, z_vals)
    Y = np.zeros_like(X)
    F = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            point = (X[i,j], Y[i,j], Z[i,j])
            F[i,j] = evaluate_dg2_quadric(coeffs, point)
    
    # Fill the region based on sign if provided
    if sign is not None:
        if axis is None:
            if sign > 0:
                plt.contourf(X, Z, F, levels=[0, F.max()], colors='red', alpha=0.1)
            else:
                plt.contourf(X, Z, F, levels=[F.min(), 0], colors='red', alpha=0.1)
        else:
            if sign > 0:
                axis.contourf(X, Z, F, levels=[0, F.max()], colors='red', alpha=0.1)
            else:
                axis.contourf(X, Z, F, levels=[F.min(), 0], colors='red', alpha=0.1)
    
    # return the contour to be plotted using matplotlib
    if axis is None:
        return plt.contour(X, Z, F, levels=[0], colors='red', alpha=0.7, linestyles='dashed')
    else:
        return axis.contour(X, Z, F, levels=[0], colors='red', alpha=0.7, linestyles='dashed')


def load_triangulation(filename):
    """
    Load a matplotlib.tri.Triangulation object from a file.
    
    Parameters:
    -----------
    filename : str
        Path to the input file
        
    Returns:
    --------
    matplotlib.tri.Triangulation
        The loaded triangulation object
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    triang = mtri.Triangulation(data['x'], data['y'], data['triangles'])
    
    if data['mask'] is not None:
        triang.set_mask(data['mask'])
    
    print(f"Triangulation loaded from {filename}")
    return triang

def save_triangulation(triang, filename):
    """
    Save a matplotlib.tri.Triangulation object to a file.
    
    Parameters:
    -----------
    triang : matplotlib.tri.Triangulation
        The triangulation object to save
    filename : str
        Path to the output file (e.g., 'mesh.tri' or 'mesh.pkl')
    """
    data = {
        'x': triang.x,
        'y': triang.y,
        'triangles': triang.triangles,
        'mask': triang.mask if triang.mask is not None else None
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Triangulation saved to {filename}")


def mtri_to_gmsh(triang, filename, model_name="mesh_from_triangulation"):
    """
    Create a GMSH mesh file from a matplotlib.tri.Triangulation object.
    
    Parameters:
    -----------
    triang : matplotlib.tri.Triangulation
        The triangulation object to convert
    filename : str
        Output mesh filename (e.g., 'output.msh')
    model_name : str, optional
        Name for the GMSH model (default: "mesh_from_triangulation")
    """
    gmsh.initialize()
    gmsh.model.add(model_name)
    
    # Create a discrete surface entity to hold the mesh
    entity_tag = gmsh.model.addDiscreteEntity(2)
    
    # Prepare node data
    n_nodes = len(triang.x)
    node_tags = list(range(1, n_nodes + 1))
    node_coords_flat = []
    for i in range(n_nodes):
        node_coords_flat.extend([triang.x[i], triang.y[i], 0])
    
    # Add all nodes at once
    gmsh.model.mesh.addNodes(2, entity_tag, node_tags, node_coords_flat)
    
    # Prepare element data (element type 2 is triangle)
    n_elements = len(triang.triangles)
    element_tags = list(range(1, n_elements + 1))
    element_connectivity = []
    for triangle in triang.triangles:
        # GMSH uses 1-based indexing, matplotlib uses 0-based
        element_connectivity.extend([triangle[0] + 1, triangle[1] + 1, triangle[2] + 1])
    
    # Add all triangular elements at once
    gmsh.model.mesh.addElementsByType(entity_tag, 2, element_tags, element_connectivity)
    
    # Set mesh format version and write
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    gmsh.write(filename)
    gmsh.finalize()
    
    print(f"GMSH mesh written to {filename}")
    
    
def create_unique_triangulation(triangulations, tol=0.001):
    """
    Create a matplotlib.tri.Triangulation object with all unique points and triangles.
    
    Parameters:
    -----------
    triangulations : list of dict
        List of triangulation blocks
    tol : float
        Tolerance for comparing point coordinates (default: 0.001)
        
    Returns:
    --------
    matplotlib.tri.Triangulation
        Triangulation object containing all unique points and triangles
    """
    import matplotlib.tri as mtri
    
    unique_points = []
    unique_triangles = []
    
    # Process each triangulation block
    for tri_data in triangulations:
        points = np.array(tri_data['points'])
        triangles = tri_data['triangles']
        
        # Map old indices to new indices in the unique points list
        index_map = {}
        
        # Process each point in this block
        for old_idx, point in enumerate(points):
            # Check if this point already exists in unique_points
            found_idx = -1
            for unique_idx, unique_point in enumerate(unique_points):
                distance = np.sqrt(np.sum((np.array(unique_point) - point)**2))
                if distance < tol:
                    found_idx = unique_idx
                    break
            
            # If point is new, add it
            if found_idx == -1:
                found_idx = len(unique_points)
                unique_points.append(tuple(point))
            
            index_map[old_idx] = found_idx
        
        # Process each triangle in this block
        for triangle in triangles:
            # Remap triangle indices to unique point indices
            new_triangle = [index_map[triangle[0]], index_map[triangle[1]], index_map[triangle[2]]]
            
            # Check if this triangle already exists (compare as sorted tuples)
            sorted_new_tri = tuple(sorted(new_triangle))
            is_duplicate = False
            for existing_tri in unique_triangles:
                if tuple(sorted(existing_tri)) == sorted_new_tri:
                    is_duplicate = True
                    break
            
            # Add triangle if it's unique
            if not is_duplicate:
                unique_triangles.append(new_triangle)
    
    # Convert to numpy arrays
    unique_points = np.array(unique_points)
    unique_triangles = np.array(unique_triangles)
    
    # Extract x and y coordinates
    x = unique_points[:, 0]
    y = unique_points[:, 1]
    
    # Create matplotlib triangulation object
    tri_mesh = mtri.Triangulation(x, y, unique_triangles)
    
    return tri_mesh

# degas2 gen_cones function
def gen_cones_all(x0,x1):
    """
    Generate quadric coefficients for a set of surfaces defined by points x0 and x1.
    Copied from `degas2` scripts.
    
    :param x0: 2D array of shape (Nsurfs, 2) representing the first point on each surface/edge
    :param x1: 2D array of shape (Nsurfs, 2) representing the second point on each surface/edge
    
    :return: 2D array of shape (Nsurfs, 10) containing the quadric coefficients for each surface
    Coefficients are in the order: [c0, cx, cy, cz, cxx, cyy, czz, cxy, cyz, cxz]
    """
    Nsurfs = np.shape(x1)[0] 
    print("Nsurfs:", Nsurfs)
    coeffs = np.zeros([Nsurfs,10])

    eps = 1.0e-8
    eps_angle = np.sqrt(2.0*1.0e-10)

    cylcond = (np.abs(x1[:,0]-x0[:,0]) < eps_angle*np.abs(x1[:,1]-x0[:,1])) 
    cylcond = np.logical_or(cylcond, (np.abs(x1[:,0]-x0[:,0]) < eps))

    planecond = (np.abs(x1[:,1]-x0[:,1]) < eps_angle*np.abs(x1[:,0]-x0[:,0])) 
    planecond = np.logical_or(planecond, (np.abs(x1[:,1]-x0[:,1]) < eps))

    m = np.divide((x1[:,1]-x0[:,1]),(x1[:,0]-x0[:,0]),out=np.ones_like(x1[:,0]),where=np.logical_not(cylcond))
    # m=1 for cylinder, slope for a cone, and 0 for a plane
    m2 = np.where(planecond, np.zeros_like(m), m*m)
    m2 = np.where(cylcond, np.ones_like(m), m2)
    # b = 0 for a cylinder, 1/2 for a plane, and intercept for a cone
    b = np.where(cylcond, np.zeros_like(m), x1[:,1] - m*x1[:,0])
    b = np.where(planecond, 0.5*np.ones_like(m), b)

    # c0 is -b^2 for a cone, -R^2 for a cylinder, -Z0 for a plane
    c0 = np.where( cylcond , -x0[:,0]**2, -b**2)
    c0 = np.where( planecond, -x0[:,1], c0)

    coeffs[:,0] = c0
    coeffs[:,3] = 2.0*b
    coeffs[:,4] = m2
    coeffs[:,5] = m2
    coeffs[:,6] = np.where(np.logical_or(planecond,cylcond),np.zeros(Nsurfs),-np.ones(Nsurfs))

    return coeffs


def is_same_quadratic(c1, c2, tol=1e-10, print_norm=False):
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)

    # Handle zero vector edge case
    if np.allclose(c1, 0, atol=tol) or np.allclose(c2, 0, atol=tol):
        return np.allclose(c1, c2, atol=tol)

    # Normalize both by their largest-magnitude coefficient
    i1 = np.argmax(np.abs(c1))
    i2 = np.argmax(np.abs(c2))

    n1 = c1 / c1[i1]
    n2 = c2 / c2[i2]
    
    if print_norm:
        print(f"Normalized {n1} and {n2}")

    return np.allclose(n1, n2, atol=tol, rtol=tol) or np.allclose(n1, -n2, atol=tol, rtol=tol)