from dolfin import Mesh, MeshEditor, refine, CellFunction, Point, cells, \
        plot, Function, FunctionSpace

def customMesh(x_i, x_f, N, sensitivePoints, nRefinement) :
    """
    N = number of intervals
    x_i = coordinate of starting point
    x_f = coordinate of end point
    sensitivePoints = array of coordinates of sensitive points
    sensRadii = array of the distance from sensitive points in which the mesh has to be refined
    """
    deltax = (x_f - x_i)/N      # Calculate equispacing
    
    mesh = Mesh()               # initialise an empty mesh
    meshEdit = MeshEditor()
    meshEdit.open(mesh, 1, 1)   # create 1D mesh
    meshEdit.init_vertices(N+1) # with N+1 vertices
    meshEdit.init_cells(N)      # and N cells
    
    for ii in range(0, N+1):    # create equispaced vertices
        meshEdit.add_vertex(ii, x_i + ii*deltax)
    
    for ii in range(0, N) :     # create cells by linking vertices
        meshEdit.add_cell(ii, ii, ii+1)
    
    meshEdit.close()
    
    # Refinement of the mesh around sensitive points
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False) # initialise cells markers to False
    for ii in range(1, nRefinement+1) :
        for SensitiveP in sensitivePoints :
            coorSensP = Point(SensitiveP)
            for cell in cells(mesh) :
                p = cell.midpoint()
                if p.distance(coorSensP) < deltax*ii :
                    cell_markers[cell] = True
        mesh = refine(mesh, cell_markers) # refine the mesh
        cell_markers = CellFunction("bool", mesh)
        cell_markers.set_all(False) # reset markers for next iteration of refinement

    
    return mesh

#mesh = customMesh(1.0, 11.0, 10, [1.0, 7.5], 2)
#U = FunctionSpace(mesh, "Lagrange", 1)
#u = Function(U)
#f = u + mesh.coordinates()
#print mesh.coordinates()
#print mesh.domains()
