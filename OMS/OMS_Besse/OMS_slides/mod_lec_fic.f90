module mod_lec_fic
  implicit none

  type msh
     REAL (kind=8), DIMENSION(3) :: MIN,MAX
     INTEGER :: nbNod,nbElm,nbPoints,nbLines,nbTriangles,nbQuads,nbTets,nbHexas,nbPrisms,nbPyramids
     REAL (kind=8), DIMENSION(:,:), allocatable :: POS
     INTEGER, DIMENSION(:,:), allocatable :: ELE_INFOS, POINTS, LINES, TRIANGLES, QUADS
     INTEGER, DIMENSION(:,:), allocatable :: TETS, HEXAS, PRISMS, PYRAMIDS
     INTEGER, DIMENSION(:,:), allocatable :: ELE_TAGS
  end type msh

contains

  subroutine fgetl(unit_nb,msg,length,ios)
    implicit none
    integer, intent(in) :: unit_nb
    integer, intent(out) :: ios,length
    character(len=256), intent(out) :: msg
    character(len=256) :: message
    character :: c
    integer :: k
    c='a'
    k=1
    ios=0

    do while ((.not.(is_iostat_eor(Ios))).AND.(.not.(is_iostat_end(Ios))))
       read (unit=unit_nb, fmt="(a)", advance="no", iostat=ios, iomsg=message) c
       msg(k:k)=c
       k=k+1
    end do
    length=k-2
  end subroutine fgetl

  subroutine load_gmsh(filename,mesh)
    ! Reads a mesh in msh format, version 1 or 2
    
    ! Copyright (C) 11/2011 C. Besse (Christophe.Besse@univ-lille1.fr)

    ! Based on load_gmsh supplied with gmsh-2.5 from
    ! R Lorph?vre (r.lorphevre@ulg.ac.be) 
    
    ! Based on load_gmsh supplied with gmsh-2.0 and load_gmsh2 from JP
    ! Moitinho de Almeida (moitinho@civil.ist.utl.pt)
    
    ! number of nodes in function of the element type
    
    ! The format 2 don't sort the elements by reg phys but by
    ! reg-elm. If this classification is important for your program,
    ! use this (after calling this function):
    !
    ! [OldRowNumber, NewRowNumber] = sort(OldMatrix(:,SortColumn)); 
    ! NewMatrix = OldMatrix(NewRowNumber,:);
    !
    ! Change the name of OldMatrix and NewMatrix with the name of yours
    ! SortColumn by the number of the last column
    !
    implicit none
    character(len=*),intent(in) :: filename
    type(msh), intent(out) :: mesh

    character(len=256) :: tline
    integer :: endoffile,ios,fileformat,fid,i,inod,nbinfo,tags,length
    integer, dimension(19) :: NODES_PER_TYPE_OF_ELEMENT
    REAL(kind=8), dimension(3) :: X
    integer, dimension(:), allocatable :: IDS
    integer, dimension(50) ::  NODES_ELEM
    NODES_PER_TYPE_OF_ELEMENT = (/2,3,4,4,8,6,5,3,6,9,10,27,18,14,1,8,20,15,13/)
    
    !print *,'Entree dans load_gmsh'
    fid=100
    open (fid,file=filename,status='unknown')
    do while (.true.)
       endoffile=0
       do while (.true.)
          call fgetl(fid,tline,length,ios)
          !print *,'TLINE = -------> ',tline(1:length),' <-------'
          if (is_iostat_end(Ios)) then
             endoffile=1
             exit
          end if
          if (tline(1:1).eq.'$' ) then
             if ((tline(2:2).eq.'N').and.(tline(3:3).eq.'O')) then
                fileformat = 1
                exit
             end if
             if ((tline(2:2).eq.'M').and.(tline(3:3).eq.'e')) then
                fileformat = 2
                call fgetl(fid,tline,length,ios);
                !print *,'TLINE = -------> ',tline(1:length),' <-------'
                print *,'Mesh Type'
                print *, tline(1:length)
                call fgetl(fid,tline,length,ios);
                if ((tline(1:1).eq.'$').and.(tline(2:2).eq.'E').and.(tline(3:3).eq.'n')) then
                   call fgetl(fid,tline,length,ios)
                   exit
                else
                   print *,' This program can only read ASCII mesh file'
                   print *,' of format 1 or 2 from GMSH, try again?'
                   endoffile=1
                   exit
                end if
             end if
             if ((tline(2:2).eq.'E').and.((tline(3:3).eq.'L').or.(tline(3:3).eq.'l'))) then
                exit
             end if
          end if
       end do
       if (endoffile.eq.1) then
          exit
       end if

       if ((tline(2:2).eq.'N').and.( (tline(3:3).eq.'O') .or. (tline(3:3).eq.'o') )) then
          print *,'reading nodes'
          call read_1_int(fid,mesh%nbNod,ios)
          !print *, mesh%nbNod
          allocate(mesh%POS(mesh%nbNod,3))
          allocate(IDS(mesh%nbNod))
          mesh.POS = 0.
          do I=1,mesh%nbNod
             call read_1_int(fid,iNod,ios)
             call read_n_real(fid,3,X,ios)
             !write(6,'(1i,3f8.4)') iNod,X
             ! VERIFIER LA TAILLE DE IDS
             IDS(iNod) = I;


             if (I.eq.1) then
                mesh%MIN = X
                mesh%MAX = X
             else
                if (mesh%MAX(1) < X(1)) mesh%MAX(1) = X(1)
                if (mesh%MAX(2) < X(2)) mesh%MAX(2) = X(2)
                if (mesh%MAX(3) < X(3)) mesh%MAX(3) = X(3)
                if (mesh%MIN(1) > X(1)) mesh%MIN(1) = X(1)
                if (mesh%MIN(2) > X(2)) mesh%MIN(2) = X(2)
                if (mesh%MIN(3) > X(3)) mesh%MIN(3) = X(3)
             end if
             mesh%POS(I, 1) = X(1)
             mesh%POS(I, 2) = X(2)
             mesh%POS(I, 3) = X(3)
          end do
          call fgetl(fid,tline,length,ios)
          print *,'nodes have been read'
       elseif ((tline(2:2).eq.'E').and.( (tline(3:3).eq.'L') .or. (tline(3:3).eq.'l'))) then
          print *,'reading elements'
          call read_1_int(fid,mesh%nbElm,ios)
          !print *,'Nb of elements ',mesh%nbElm
          if (fileformat.eq.1) then
             nbinfo = 5;
             tags = 3;
          end if
          if (fileformat.eq.2) then
             nbinfo = 4;
             tags = 4;
          end if
          allocate(mesh%ELE_INFOS(mesh%nbElm,nbinfo))
          mesh%nbPoints = 0
          mesh%nbLines = 0
          mesh%nbTriangles = 0
          mesh%nbQuads = 0
          mesh%nbTets = 0
          mesh%nbHexas = 0
          mesh%nbPrisms = 0
          mesh%nbPyramids = 0
          ! comment next 8 lines to get "tight" arrays (will slow down reading)
          allocate(mesh%POINTS(mesh%nbElm, 2))
          allocate(mesh%LINES(mesh%nbElm, 3))
          allocate(mesh%TRIANGLES(mesh%nbElm, 4))
          allocate(mesh%QUADS(mesh%nbElm, 5))
          allocate(mesh%TETS(mesh%nbElm, 5))
          allocate(mesh%HEXAS(mesh%nbElm, 9))
          allocate(mesh%PRISMS(mesh%nbElm, 7))
          allocate(mesh%PYRAMIDS(mesh%nbElm, 6))
          allocate(mesh%ELE_TAGS(mesh%nbElm, 10))
          do I = 1,mesh%nbElm
             call read_n_int(fid,nbinfo,mesh%ELE_INFOS(I, :),ios)
             if (fileformat.eq.1) then
                ! take the mesh%ELE_INFOS(I, 5) nodes of the element
                call read_n_int(fid,mesh%ELE_INFOS(I, 5),NODES_ELEM,ios)
          end if
          if (fileformat.eq.2) then
             call read_n_int(fid,(mesh%ELE_INFOS(I,3)-1),mesh%ELE_TAGS(I,:),ios)
             ! take the msh.NODES_PER_TYPE_OF_ELEMENT (mesh%ELE_INFOS(I, 2)) nodes of the element
             call read_n_int(fid,NODES_PER_TYPE_OF_ELEMENT(mesh.ELE_INFOS(I, 2)),NODES_ELEM,ios)
          end if
          if(mesh%ELE_INFOS(I, 2).eq.15) then !! point
             mesh%nbPoints = mesh%nbPoints + 1
             mesh%POINTS(mesh%nbPoints, 1) = IDS(NODES_ELEM(1));
             mesh%POINTS(mesh%nbPoints, 2) = mesh%ELE_INFOS(I, tags);
          end if
          if(mesh%ELE_INFOS(I, 2).eq.1) then !! line
             mesh%nbLines = mesh%nbLines + 1;
             mesh%LINES(mesh%nbLines, 1) = IDS(NODES_ELEM(1));
             mesh%LINES(mesh%nbLines, 2) = IDS(NODES_ELEM(2));   
             mesh%LINES(mesh%nbLines, 3) = mesh%ELE_INFOS(I, tags);
          end if
          if(mesh%ELE_INFOS(I, 2).eq.2) then !! triangle
             mesh%nbTriangles = mesh%nbTriangles + 1;
             mesh%TRIANGLES(mesh%nbTriangles, 1) = IDS(NODES_ELEM(1));
             mesh%TRIANGLES(mesh%nbTriangles, 2) = IDS(NODES_ELEM(2));
             mesh%TRIANGLES(mesh%nbTriangles, 3) = IDS(NODES_ELEM(3));
             mesh%TRIANGLES(mesh%nbTriangles, 4) = mesh%ELE_INFOS(I, tags);
          end if
          if(mesh%ELE_INFOS(I, 2).eq.3) then !! quadrangle
             mesh%nbQuads = mesh%nbQuads + 1;
             mesh%QUADS(mesh%nbQuads, 1) = IDS(NODES_ELEM(1));
             mesh%QUADS(mesh%nbQuads, 2) = IDS(NODES_ELEM(2));
             mesh%QUADS(mesh%nbQuads, 3) = IDS(NODES_ELEM(3));
             mesh%QUADS(mesh%nbQuads, 4) = IDS(NODES_ELEM(4));
             mesh%QUADS(mesh%nbQuads, 5) = mesh%ELE_INFOS(I, tags);
          end if
          if(mesh%ELE_INFOS(I, 2).eq.4) then !! tetrahedron
             mesh%nbTets = mesh%nbTets + 1;
             mesh%TETS(mesh%nbTets, 1) = IDS(NODES_ELEM(1));
             mesh%TETS(mesh%nbTets, 2) = IDS(NODES_ELEM(2));
             mesh%TETS(mesh%nbTets, 3) = IDS(NODES_ELEM(3));
             mesh%TETS(mesh%nbTets, 4) = IDS(NODES_ELEM(4));
             mesh%TETS(mesh%nbTets, 5) = mesh%ELE_INFOS(I, tags);
          end if
          if(mesh%ELE_INFOS(I, 2).eq.5) then !! hexahedron
             mesh%nbHexas = mesh%nbHexas + 1;
             mesh%HEXAS(mesh%nbHexas, 1) = IDS(NODES_ELEM(1));
             mesh%HEXAS(mesh%nbHexas, 2) = IDS(NODES_ELEM(2));
             mesh%HEXAS(mesh%nbHexas, 3) = IDS(NODES_ELEM(3));
             mesh%HEXAS(mesh%nbHexas, 4) = IDS(NODES_ELEM(4));
             mesh%HEXAS(mesh%nbHexas, 5) = IDS(NODES_ELEM(5));
             mesh%HEXAS(mesh%nbHexas, 6) = IDS(NODES_ELEM(6));
             mesh%HEXAS(mesh%nbHexas, 7) = IDS(NODES_ELEM(7));
             mesh%HEXAS(mesh%nbHexas, 8) = IDS(NODES_ELEM(8));
             mesh%HEXAS(mesh%nbHexas, 9) = mesh%ELE_INFOS(I, tags);
          end if
          if(mesh%ELE_INFOS(I, 2).eq.6) then !! prism
             mesh%nbPrisms = mesh%nbPrisms + 1;
             mesh%PRISMS(mesh%nbPrisms, 1) = IDS(NODES_ELEM(1));
             mesh%PRISMS(mesh%nbPrisms, 2) = IDS(NODES_ELEM(2));
             mesh%PRISMS(mesh%nbPrisms, 3) = IDS(NODES_ELEM(3));
             mesh%PRISMS(mesh%nbPrisms, 4) = IDS(NODES_ELEM(4));
             mesh%PRISMS(mesh%nbPrisms, 5) = IDS(NODES_ELEM(5));
             mesh%PRISMS(mesh%nbPrisms, 6) = IDS(NODES_ELEM(6));
             mesh%PRISMS(mesh%nbPrisms, 7) = mesh%ELE_INFOS(I, tags);
          end if
          if(mesh%ELE_INFOS(I, 2).eq.7) then !! pyramid
             mesh%nbPyramids = mesh%nbPyramids + 1;
             mesh%PYRAMIDS(mesh%nbPyramids, 1) = IDS(NODES_ELEM(1));
             mesh%PYRAMIDS(mesh%nbPyramids, 2) = IDS(NODES_ELEM(2));
             mesh%PYRAMIDS(mesh%nbPyramids, 3) = IDS(NODES_ELEM(3));
             mesh%PYRAMIDS(mesh%nbPyramids, 4) = IDS(NODES_ELEM(4));
             mesh%PYRAMIDS(mesh%nbPyramids, 5) = IDS(NODES_ELEM(5));
             mesh%PYRAMIDS(mesh%nbPyramids, 6) = IDS(NODES_ELEM(6));
             mesh%PYRAMIDS(mesh%nbPyramids, 7) = mesh%ELE_INFOS(I, tags);
          end if
       end do
       call fgetl(fid,tline,length,ios)
    print *,'elements have been read'
    
 end if
end do

close(fid)
end subroutine load_gmsh
  

  subroutine read_1_int(unit_nb,num,ios)
    implicit none
    integer, intent(out) :: num,ios
    integer, intent(in) :: unit_nb
    character(len=256) :: msg,message
    character :: c
    integer :: k
    c='a'
    k=1
    ios=0
    do while ((c.ne.' ').and.(.not.(is_iostat_eor(ios))))
       read (unit=unit_nb, fmt="(a)", advance="no", iostat=ios, iomsg=message) c
       msg(k:k)=c
!       print *,'debug read_1_int ',c
       k=k+1
    end do
    read(msg,*) num
  end subroutine read_1_int

  subroutine read_n_int(unit_nb,n,num,ios)
    implicit none
    integer, intent(out), dimension(:) :: num
    integer, intent(out) :: ios
    integer, intent(in) :: unit_nb,n
    character(len=256) :: msg,message
    character :: c
    integer :: k,j
    ios=0
    do j=1,n
       c='a'
       k=1
       do while ((c.ne.' ').and.(.not.(is_iostat_eor(ios))))
          read (unit=unit_nb, fmt="(a)", advance="no", iostat=ios, iomsg=message) c
          msg(k:k)=c
!          print *,'debug read_n_int ',c
          k=k+1
       end do
       read(msg,*) num(j)
    end do


  end subroutine read_n_int

  subroutine read_1_real(unit_nb,num,ios)
    implicit none
    real (kind=8), intent(out) :: num
    integer, intent(out) :: ios
    integer, intent(in) :: unit_nb
    character(len=256) :: msg,message
    character :: c
    integer :: k
    c='a'
    k=1
    ios=0
    do while ((c.ne.' ').and.(.not.(is_iostat_eor(ios))))
       read (unit=unit_nb, fmt="(a)", advance="no", iostat=ios, iomsg=message) c
       msg(k:k)=c
!       print *,'debug read_1_real ',c
       k=k+1
    end do
    read(msg,*) num
  end subroutine read_1_real

  subroutine read_n_real(unit_nb,n,num,ios)
    implicit none
    real (kind=8), intent(out), dimension(:) :: num
    integer, intent(out) :: ios
    integer, intent(in) :: unit_nb,n
    character(len=256) :: msg,message
    character :: c
    integer :: k,j
    ios=0
    do j=1,n
       c='a'
       k=1
       do while ((c.ne.' ').and.(.not.(is_iostat_eor(ios))))
          read (unit=unit_nb, fmt="(a)", advance="no", iostat=ios, iomsg=message) c
          msg(k:k)=c
!          print *,'debug read_n_real ',c          
          k=k+1
       end do
       read(msg,*) num(j)
    end do
  end subroutine read_n_real

  integer function strlen(st)
    integer		i
    character		st*(*)
    i = len(st)
    do while (st(i:i) .eq. ' ')
       i = i - 1
    enddo
    strlen = i
    return
  end function strlen
end module mod_lec_fic
