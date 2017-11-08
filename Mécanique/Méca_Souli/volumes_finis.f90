program volumes_finis

  parameter (nx=3, ny=3, n=nx*ny)
  dimension vois(4,n), a(n,n), b(n), u(n)
  integer vois, k1, k2, k3, k4

! Remplir la matrice des voisins
do i = 1, nx
  do j = 1, ny
    k = (i-1)*nx + j
    vois(1,k) = k - 1
    vois(2,k) = k + 1
    vois(3,k) = k - nx
    vois(4,k) = k + nx
  end do
end do

! Corriger les voisins
  ! sur Gamma_1 :
do j = 1, ny
  k1 = (j-1)*nx + 1
  vois(1,k1) = -k1
end do

  ! sur Gamma_2 :
do j = 1, ny
  k2 = j*nx
  vois(2,k2) = -k2
end do

  ! sur Gamma_3 :
do i = 1, nx
  k3 = i
  vois(3,k3) = -k3
end do

  ! sur Gamma_4 :
do i = 1, nx
  k4 = nx*(ny-1) + i
  vois(4,k4) = -k4
end do

! Initialisation de la matrice a
  a(:,:) = 0

! Matrice d'assemblage a
k = 0
do j = 1, ny
  do i = 1, nx
    k = k + 1                    ! k = (j-1)*nx + i

    if (vois(1,k) .gt. 0) then
      k1 = (j-1)*nx + 1
      a(k,k)  = a(k,k)  + 1
      a(k,k1) = a(k,k1) - 1
    end if

    if (vois(2,k) .gt. 0) then
      k2 = j*nx
      a(k,k)  = a(k,k)  + 1
      a(k,k2) = a(k,k2) - 1
    end if

    if (vois(3,k) .gt. 0) then
      k3 = i
      a(k,k)  = a(k,k)  + 1
      a(k,k3) = a(k,k3) - 1
    end if

    if (vois(4,k) .gt. 0) then
      k4 = nx*(ny-1) + i
      a(k,k)  = a(k,k)  + 1
      a(k,k4) = a(k,k4) - 1
    end if

    write(*,*) k, (vois(l,k), l=1,4)

  end do
end do

! Affichage de a
print*, "a ="
do j = 1, n
  write(*,*) (a(i,j), i=1,n)
end do



end program volumes_finis
