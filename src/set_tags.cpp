/**
* Set mesh tags manually to recreate the degas2 geometry
* It will be used in the test case for readOH2CSG
*/
#include <iostream>
#include <stdexcept>
#include <string>

#include <Omega_h_for.hpp>
#include <Omega_h_mark.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_array.hpp>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("ERROR: Provided %d arguments:", argc-1);
        for (int i = 0; i < argc; ++i) {
            printf(" %s", argv[i]);
        }
        printf("\nUsage: %s <inputmesh.osh> <outputmesh.osh>\n", argv[0]);
        return 1;
    }

    std::string input_mesh_name = argv[1];
    std::string output_mesh_name = argv[2];


    auto lib = Omega_h::Library(&argc, &argv);
    const auto world = lib.world();
    Omega_h::Mesh input_mesh = Omega_h::binary::read(input_mesh_name, world);

    printf("Read Mesh with %d elements\n", input_mesh.nfaces());
    const int nfaces = input_mesh.nfaces();
    const int nedges = input_mesh.nedges();
    const int nnodes = input_mesh.nverts();

    Omega_h::HostWrite<int> wall_emb_tag(nfaces);
    
    for (int i=0; i<nfaces; ++i) {
        if (i < 12) {
            wall_emb_tag[i] = 0;
        } else {
            wall_emb_tag[i] = 1;
        }

    }

    Omega_h::Write wall_emb_tag_read(wall_emb_tag);
    input_mesh.add_tag<int>(Omega_h::FACE, "offset_face", 1);
    input_mesh.set_tag(2, "offset_face", Omega_h::Read(wall_emb_tag_read));


    Omega_h::HostWrite<int> wall_node_tag(nnodes);
    // Fill all elements with 0
    for (int i = 0; i < nnodes; ++i) {
        wall_node_tag[i] = 0;
    }

    // Set specific indices to 1
    int indices_to_set[] = {4, 6, 10, 8, 9, 5, 2, 1};
    for (int idx : indices_to_set) {
        if (idx >= 0 && idx < nnodes) {
            wall_node_tag[idx] = 1;
        }
    }

    Omega_h::Write wall_node_tag_read(wall_node_tag);
    input_mesh.add_tag<int>(Omega_h::VERT, "isOnWall", 1);
    input_mesh.set_tag(0, "isOnWall", Omega_h::Read(wall_node_tag_read));

    printf("Writing mesh to files\n");
    Omega_h::binary::write(output_mesh_name, &input_mesh);
    Omega_h::vtk::write_parallel(output_mesh_name+".vtk", &input_mesh, input_mesh.dim());

    return 0;
}