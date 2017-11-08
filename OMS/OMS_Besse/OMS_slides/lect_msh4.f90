program lect_msh4

  use mod_lec_fic
  implicit none
  type(msh) :: mesh
  integer :: j

  call load_gmsh('carre.msh',mesh)
  
  print *,"Nombre de noeuds ",mesh%nbNod
  do j=1,mesh%nbNod
     write(6,'(1i,3f8.4)') j,mesh%pos(j,:)
  end do
  print *,"Nombre de triangles",mesh%nbTriangles
  do j=1,mesh%nbTriangles
     write(6,'(5i)') j,mesh%TRIANGLES(j,:)
  end do

end program lect_msh4
