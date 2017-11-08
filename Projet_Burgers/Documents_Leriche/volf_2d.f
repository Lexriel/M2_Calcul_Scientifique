	program volf_2d
        parameter (nx=3,ny=3,n=nx*ny)
        dimension a(n,n),b(n),x(n),ivois(4,n),t0(n),t1(n)
c calcul voisin
        do j=1,ny
        do i=1,nx
         k= (j-1)*nx+i
          ivois(1,k)=k-1
          ivois(2,k)=k+1
          ivois(3,k)=k-nx
          ivois(4,k)=k+nx
        enddo
        enddo
c voisin d'en bas  le 3 est faux
          do k=1,nx
           ivois(3,k)=-k
          enddo
c voisin   en haut le 4 est faux
          do k=1,nx
           k1=n-nx+k
           ivois(4,k1)=-k1
          enddo
c voisin a gauche le 1 faux
          do k=1,ny
           k1=(k-1)*nx+1
           ivois(1,k1)=-k1
          enddo
c voisin  a droite 2 est faux
          do k=1,ny
           k1=k*nx
           ivois(2,k1)=-k1
          enddo
         do k=1,n
          write(* ,*)k,(ivois(i,k),i=1,4)
         enddo
         do i=1,n
          b(i)=0.
         do j=1,n
           a(i,j)=0.
         enddo
         enddo
          h=1.
         do k=1,n
          k1=ivois(1,k)
          k2=ivois(2,k)
          k3=ivois(3,k)
          k4=ivois(4,k)
          if(k1.gt.0)then
             a(k,k1)=a(k,k1)-1.
             a(k,k)=a(k,k)+1
          endif
          if(k2.gt.0)then
             a(k,k2)=a(k,k2)-1.
             a(k,k)=a(k,k)+1.
          endif
          if(k3.gt.0)then
             a(k,k3)=a(k,k3)-1.
             a(k,k)=a(k,k)+1.
          endif
          if(k4.gt.0)then
             a(k,k4)=a(k,k4)-1.
             a(k,k)=a(k,k)+1.
          endif
         enddo
         do k=1,n
          write(* ,'(9f6.3)')(a(k,i),i=1,n)
         enddo
         stop
c conditions aux limites a droite
         do i=1,ny
          k=(i-1)*nx+1
          do j=1,n
           a(k,j)=0.
          enddo
          a(k,k)=1.
          b(k)=10.
         enddo
c conditions au x limites a gauche
         do i=1,ny
          k=i*nx
          do j=1,n
           a(k,j)=0.
          enddo
          a(k,k)=1.
          b(k)=20.
         enddo
c resolution par methode iterative de Jacobi
         do i=1,n
          t0(i)=0.
         enddo
         do i=1,ny
          k=(i-1)*nx+1
          t0(k)=10.
         enddo 
         do i=1,ny
          k=i*nx
          t0(k)=20.
         enddo 
         iter=0
  1      do i=1,n
           som1=0.
           do j=1,i-1
            som1=som1+a(i,j)*t0(j)
           enddo
           som2=0.
           do j=i+1,n
            som2=som2+a(i,j)*t0(j)
           enddo
           t1(i)=(b(i)-som1-som2)/a(i,i)
          enddo
           er=0.
           do i=1,n
            er=er+abs(t1(i)-t0(i))
           enddo
          if(er.gt.1.e-7)then
           do i=1,n
            t0(i)=t1(i)
           enddo
           iter=iter+1
           write(*,*)'iter=',iter,er
           pause
           goto 1
          endif
          do i=1,nx
            k=ny/2*nx+i
            xi=(i-1)*h
            write(1,*)xi,t1(k)
          enddo
         stop
         end
