module methode
contains
  subroutine diacsr (n,job,idiag,diag,ndiag,ioff,a,ja,ia)
    integer n,job,idiag,ndiag
    real*8 diag(n,idiag), a(*), t
    integer ia(*), ja(*), ioff(*)
    !----------------------------------------------------------------------- 
    !    diagonal format     to     compressed sparse row     
    !----------------------------------------------------------------------- 
    ! this subroutine extract the idiag most important diagonals from the 
    ! input matrix a, ja, ia, i.e, those diagonals of the matrix which have
    ! the largest number of nonzero elements. If requested (see job),
    ! the rest of the matrix is put in a the output matrix ao, jao, iao
    !----------------------------------------------------------------------- 
    ! on entry:
    !---------- 
    ! n	= integer. dimension of the matrix a.
    ! job	= integer. job indicator with the following meaning.
    !         if (job .eq. 0) then check for each entry in diag
    !         whether this entry is zero. If it is then do not include
    !         in the output matrix. Note that the test is a test for
    !         an exact arithmetic zero. Be sure that the zeros are
    !         actual zeros in double precision otherwise this would not
    !         work.
    !         
    ! idiag = integer equal to the number of diagonals to be extracted. 
    !         Note: on return idiag may be modified.
    !
    ! diag  = real array of size (ndiag x idiag) containing the diagonals
    !         of A on return. 
    ! 
    ! ndiag = integer equal to the first dimension of array diag.
    !
    ! ioff  = integer array of length idiag, containing the offsets of the
    !   	  diagonals to be extracted.
    !
    ! on return:
    !----------- 
    ! a, 
    ! ja, 			
    ! ia    = matrix stored in a, ja, ia, format
    !
    ! Note:
    ! ----- the arrays a and ja should be of length n*idiag.
    !
    !----------------------------------------------------------------------- 
    integer ko,i,jj,j

    ia(1) = 1
    ko = 1
    do i=1, n
       loop70 : do jj = 1, idiag
          j = i+ioff(jj) 
          if (j .lt. 1 .or. j .gt. n) cycle loop70
          t = diag(i,jj) 
          if (job .eq. 0 .and. t .eq. 0.0d0) cycle loop70
          a(ko) = t
          ja(ko) = j
          ko = ko+1
       end do loop70
       ia(i+1) = ko
    end do
    return
    !----------- end of diacsr ---------------------------------------------
    !-----------------------------------------------------------------------
  end subroutine diacsr


  FUNCTION cbcg(A,IA,JA,b,eps,itermax,BILUT,ibilut,jbilut,debug) RESULT(x)
    USE BLAS95
    IMPLICIT NONE
    !---------------------------------------------------
    ! Gradient conjugue reel pour resolution du
    ! systeme lineaire reel Ax=b symetrique
    ! Christophe Besse 2010
    ! Variables d'entree   : matrice A
    !                        2nd membre b
    !                        precision eps (OPTION)
    !                        nb iter maxi itermax (OPTION)
    !                        M matr4ice de PRECOND (OPTION)
    !                        ju matrice pour le PRECOND (OPTION)
    !                        debug (OPTION)
    !
    ! Variables de sorties : resultat x
    !---------------------------------------------------
    INTEGER ,INTENT(in), DIMENSION(:) :: IA,JA
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: b
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: A
    REAL (kind=8), OPTIONAL, INTENT(in) :: eps
    INTEGER, OPTIONAL, INTENT(in) :: itermax
    INTEGER, OPTIONAL, INTENT(in) :: debug
    REAL (kind=8) , OPTIONAL, INTENT(in), DIMENSION(:) :: BILUT
    INTEGER , OPTIONAL, INTENT(in), DIMENSION(:) :: IBILUT,JBILUT
    REAL (kind=8), DIMENSION(SIZE(b)) :: x

    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: x0,rj,rj1,p,vtmp,zj,tmpdebug,yj
    REAL (kind=8) :: alpha,beta,rhoj,rhoj1
    REAL (kind=8) :: prec
    INTEGER :: n,k

    n=SIZE(b)
    ALLOCATE(x0(n),yj(n))
    ALLOCATE(rj(n))
    ALLOCATE(rj1(n))
    ALLOCATE(p(n),zj(n))
    ALLOCATE(vtmp(n))
    ALLOCATE(tmpdebug(n))

    IF (PRESENT(EPS)) THEN
       prec=eps
    ELSE
       prec = 1.e-6
    END IF

    k=0
    x0(:)=0.d0

    CALL MKL_DCSRSYMV('U', N, A, IA, JA, x0,rj)
    rj=b-rj    !rj=b-A*x0

    IF (PRESENT(debug)) THEN
       WRITE(6,*) "Debug"
    END IF

    DO
       IF (PRESENT(BILUT)) THEN              ! Preconditionnement
        call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, rj,yj)
        call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, yj,zj)
       ELSE
          zj=rj
       END IF

       rhoj1=dot(zj,rj)
       
       IF (k.EQ.0) THEN
          p=zj
       ELSE
          beta=rhoj1/rhoj
          p=zj+beta*p
       END IF

       CALL MKL_DCSRSYMV('U', N, A, IA, JA, p,vtmp)       !vtmp=A*p
       alpha=rhoj1/dot(vtmp,p)   
       x=x0+alpha*p
       rj1=rj-alpha*vtmp

       IF (MAXVAL(rj1).EQ.0.) RETURN


       k=k+1
       IF (PRESENT(itermax)) THEN
          IF ((testcv(x0,x,prec,debug)).OR.(k.GT.itermax)) THEN
             IF (PRESENT(debug)) THEN
                IF (debug.EQ.1) THEN
                   WRITE(6,*) k," iterations"
                endif
             endif
             EXIT
          END IF
       ELSE
          IF (testcv(x0,x,prec,debug)) EXIT
       END IF
       x0=x
       rj=rj1
       rhoj=rhoj1
       
    END DO

    DEALLOCATE(x0)
    DEALLOCATE(rj)
    DEALLOCATE(rj1)
    DEALLOCATE(p,zj)
    DEALLOCATE(vtmp)
    DEALLOCATE(tmpdebug)


    RETURN
  END FUNCTION cbcg
       
  FUNCTION testcv(x,y,eps,debug) RESULT(l)
    !---------------------------------------------------
    ! Test de convergence
    !---------------------------------------------------
    USE BLAS95
    IMPLICIT NONE
    REAL (kind=8), INTENT(in), DIMENSION(:) :: x,y
    REAL (kind=8), INTENT(in) :: eps
    INTEGER, OPTIONAL, INTENT(in) :: debug
    LOGICAL :: l

!!$    IF (PRESENT(debug)) THEN
!!$       IF (debug.EQ.1) THEN
!!$          WRITE(6,*) "Erreur relative : ",ABS(MAXVAL(y-x)/MAXVAL(x))
!!$       END IF
!!$    END IF
!!$    l=(ABS(MAXVAL(y-x)/MAXVAL(x))).LT.eps
    IF (PRESENT(debug)) THEN
       IF (debug.EQ.1) THEN
          WRITE(6,*) "Erreur relative : ",nrm2(y-x)/nrm2(x)
       END IF
    END IF
    l=(nrm2(y-x)/nrm2(x)).LT.eps

  END FUNCTION testcv



  SUBROUTINE csr_to_full(n,iwk,A,IA,JA,F)
    ! Christophe Besse 2010
    integer, intent(in) :: n,iwk
    real (kind=8), dimension(iwk), intent(in) :: a
    integer ,dimension(iwk), intent(in) :: ja
    integer ,dimension(n+1), intent(in) :: ia
    
    real (kind=8), INTENT(out), DIMENSION(n,n) :: F
    
    INTEGER :: i,k,j
    
    F=0.0d0

    DO i=1,n
       DO k=ia(i),ia(i+1)-1
          j = ja(k) 
          F(i,j) = A(k)
       END DO
    END DO

    RETURN
  END SUBROUTINE csr_to_full


  SUBROUTINE prints(n,iwk,A,IA,JA)
    ! Christophe Besse 2010
    integer, intent(in) :: n,iwk
    real (kind=8), dimension(iwk), intent(in) :: a
    integer ,dimension(iwk), intent(in) :: ja
    integer ,dimension(n+1), intent(in) :: ia
    REAL (kind=8), DIMENSION(:,:), ALLOCATABLE :: B
    INTEGER :: ncol,j,k
    CHARACTER (len=40) :: forma

    ncol=n
    ALLOCATE(B(n,n)) 
    CALL csr_to_full(n,iwk,A,IA,JA,B)

    WRITE(FORMA,*) "(",ncol,"f9.4)"
    WRITE(6,'(a1,1x)',ADVANCE='NO') "/"
    DO k=1,ncol
       IF (B(1,k).EQ.0.0d0) THEN
          WRITE(6,'(8x,a1)',ADVANCE='NO') "0"
       ELSE
          WRITE(6,'(f9.4)',ADVANCE='NO')  B(1,k)
       END IF
    END DO
    WRITE(6,'(1x,a1,1x)') "\\"
    DO j=2,n-1
       WRITE(6,'(a1,1x)',ADVANCE='NO') "|"
       DO k=1,ncol
          IF (B(j,k).EQ.0.0d0) THEN
             WRITE(6,'(8x,a1)',ADVANCE='NO') "0"
          ELSE
             WRITE(6,'(f9.4)',ADVANCE='NO')  B(j,k)
          END IF
       END DO
       WRITE(6,'(1x,a1,1x)') "|"
    END DO
    WRITE(6,'(a1,1x)',ADVANCE='NO') "\\"
    DO k=1,ncol
       IF (B(n,k).EQ.0.0d0) THEN
          WRITE(6,'(8x,a1)',ADVANCE='NO') "0"
       ELSE
          WRITE(6,'(f9.4)',ADVANCE='NO')  B(n,k)
       END IF
    END DO
    WRITE(6,'(1x,a1,1x)') "/"
    DEALLOCATE(B)

    RETURN
  END SUBROUTINE prints


  FUNCTION bcgr(A,IA,JA,b,eps,itermax,BILUT,ibilut,jbilut,debug) RESULT(x)
    USE BLAS95
    IMPLICIT NONE
    !---------------------------------------------------
    ! Bigradient conjugue reel pour resolution du
    ! systeme lineaire reel Ax=b non symetrique
    ! Christophe Besse 2010
    ! Variables d'entree   : matrice A
    !                        2nd membre b
    !                        precision eps (OPTION)
    !                        nb iter maxi itermax (OPTION)
    !                        M matr4ice de PRECOND (OPTION)
    !                        ju matrice pour le PRECOND (OPTION)
    !                        debug (OPTION)
    !
    ! Variables de sorties : resultat x
    !---------------------------------------------------
    INTEGER ,INTENT(in), DIMENSION(:) :: IA,JA
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: b
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: A
    REAL (kind=8), OPTIONAL, INTENT(in) :: eps
    INTEGER, OPTIONAL, INTENT(in) :: itermax
    INTEGER, OPTIONAL, INTENT(in) :: debug
    REAL (kind=8) , OPTIONAL, INTENT(in), DIMENSION(:) :: BILUT
    INTEGER , OPTIONAL, INTENT(in), DIMENSION(:) :: IBILUT,JBILUT
    REAL (kind=8), DIMENSION(SIZE(b)) :: x

    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: x0,rj,sj,rj1,sj1,p,q,vtmp1,vtmp2
    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: z1j,z2j,y1j,y2j
    REAL (kind=8) :: alpha,beta,rhoj,rhoj1
    REAL (kind=8) :: prec
    INTEGER :: n,k,imax

    n=SIZE(b)
    ALLOCATE(x0(n))
    ALLOCATE(rj(n))
    ALLOCATE(sj(n))
    ALLOCATE(rj1(n))
    ALLOCATE(sj1(n))
    ALLOCATE(p(n))
    ALLOCATE(q(n),z1j(n),z2j(n),y1j(n),y2j(n))
    ALLOCATE(vtmp1(n),vtmp2(n))

    IF (PRESENT(EPS)) THEN
       prec=eps
    ELSE
       prec = 1.e-6
    END IF

    IF (PRESENT(itermax)) THEN
       imax=itermax
    ELSE
       imax=50
    END IF


    k=0
    x0(:)=0.d0

    CALL MKL_DCSRSYMV('U', N, A, IA, JA, x0,rj)
    rj=b-rj    !rj=b-A*x0
    sj=rj

    IF (PRESENT(debug)) THEN
       WRITE(6,*) "Debug"
    END IF

    DO
       IF (PRESENT(BILUT)) THEN              ! Preconditionnement
          call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, rj,y1j)    !y1j=lsol(M1,rj)  
          call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, y1j,z1j)   !z1j=usol(M2,y1j) 
          call mkl_dcsrtrsv('L','T','U', n, bilut, ibilut, jbilut, sj,y2j)    !y2j=lsol(M2t,sj) 
          call mkl_dcsrtrsv('U','T','N', n, bilut, ibilut, jbilut, y2j,z2j)   !z2j=usol(M1t,y2j)
       ELSE
          z1j=rj
          z2j=sj
       END IF

       rhoj1=dot(z1j,sj) ! rhoj1=z1j.ps.sj
       
       IF (rhoj1.EQ.0.0d0) STOP "bcgs_r : method fails"
       IF (k.EQ.0) THEN
          p=z1j
          q=z2j
       ELSE
          beta=rhoj1/rhoj
          p=z1j+beta*p
          q=z2j+beta*q
       END IF

       CALL MKL_DCSRGEMV('N', N, A, IA, JA, p,vtmp1)       !vtmp1=A*p
       CALL MKL_DCSRGEMV('T', N, A, IA, JA, q,vtmp2)       !vtmp2=trA*q
       
       alpha=rhoj1/dot(vtmp1,q)   ! Preconditionnement
       x=x0+alpha*p
       rj1=rj-alpha*vtmp1
       sj1=sj-alpha*vtmp2

       IF (MAXVAL(rj1).EQ.0.d0) RETURN

       k=k+1
       IF ((testcv(x0,x,prec,debug)).OR.(k.GT.imax)) THEN
          WRITE(6,*) k," iterations"
          EXIT
       END IF
       
       x0=x
       rj=rj1
       sj=sj1
       rhoj=rhoj1
       
    END DO

    DEALLOCATE(x0)
    DEALLOCATE(rj)
    DEALLOCATE(sj)
    DEALLOCATE(rj1)
    DEALLOCATE(sj1,z1j,z2j,y1j,y2j)
    DEALLOCATE(p)
    DEALLOCATE(q)
    DEALLOCATE(vtmp1,vtmp2)

    RETURN
  END FUNCTION bcgr



  FUNCTION cgs(A,IA,JA,b,eps,itermax,BILUT,ibilut,jbilut,debug) RESULT(x)
    USE BLAS95
    IMPLICIT NONE
    !---------------------------------------------------
    ! Conjugate Gradient Square reel pour resolution du
    ! systeme lineaire reel Ax=b non symetrique
    ! Christophe Besse 2010
    ! Variables d'entree   : matrice A
    !                        2nd membre b
    !                        precision eps (OPTION)
    !                        nb iter maxi itermax (OPTION)
    !                        M matr4ice de PRECOND (OPTION)
    !                        ju matrice pour le PRECOND (OPTION)
    !                        debug (OPTION)
    !
    ! Variables de sorties : resultat x
    !---------------------------------------------------
    INTEGER ,INTENT(in), DIMENSION(:) :: IA,JA
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: b
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: A
    REAL (kind=8), OPTIONAL, INTENT(in) :: eps
    INTEGER, OPTIONAL, INTENT(in) :: itermax
    INTEGER, OPTIONAL, INTENT(in) :: debug
    REAL (kind=8) , OPTIONAL, INTENT(in), DIMENSION(:) :: BILUT
    INTEGER , OPTIONAL, INTENT(in), DIMENSION(:) :: IBILUT,JBILUT
    REAL (kind=8), DIMENSION(SIZE(b)) :: x

    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: x0,rj,s,rj1,p,q,vtmp1,vtmp2
    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: zj,u,v,yj
    REAL (kind=8) :: alpha,beta,rhoj1,rhoj
    REAL (kind=8) :: prec
    INTEGER :: n,k,imax

    n=SIZE(b)
    ALLOCATE(x0(n))
    ALLOCATE(rj(n))
    ALLOCATE(s(n))
    ALLOCATE(rj1(n))
    ALLOCATE(p(n),u(n),v(n))
    ALLOCATE(q(n),zj(n),yj(n))
    ALLOCATE(vtmp1(n),vtmp2(n))

    IF (PRESENT(EPS)) THEN
       prec=eps
    ELSE
       prec = 1.e-6
    END IF

    IF (PRESENT(itermax)) THEN
       imax=itermax
    ELSE
       imax=50
    END IF


    k=0

    x0(:)=0.d0

    CALL MKL_DCSRSYMV('U', N, A, IA, JA, x0,rj)
    rj=b-rj    !rj=b-A*x0
    s=rj

    IF (PRESENT(debug)) THEN
       WRITE(6,*) "Debug"
    END IF


    DO
       rhoj1=dot(rj,s)
       IF (rhoj1.EQ.0.0d0) STOP "cgs : method fails"

       IF (k.EQ.0) THEN
          u=rj
          p=u
       ELSE
          beta=rhoj1/rhoj
          u=rj+beta*q
          p=u+beta*(q+beta*p)
       END IF


       IF (PRESENT(BILUT)) THEN              ! Preconditionnement
          call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, p,yj)
          call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, yj,zj)
       ELSE
          zj=p
       END IF

       CALL MKL_DCSRGEMV('N', N, A, IA, JA, zj,vtmp1)       !vtmp1=A*zj
       alpha=rhoj1/dot(vtmp1,s)   ! Preconditionnement
       q=u-alpha*vtmp1
       vtmp2=u+q
       IF (PRESENT(BILUT)) THEN              ! Preconditionnement
          call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, vtmp2,yj)
          call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, yj,v)
       ELSE
          v=vtmp2
       END IF
       x=x0+alpha*v
       CALL MKL_DCSRGEMV('N', N, A, IA, JA, v,vtmp2)       !vtmp2=A*v
       rj1=rj-alpha*vtmp2

       IF (MAXVAL(rj1).EQ.0.d0) RETURN

       k=k+1
       IF ((testcv(x0,x,prec,debug)).OR.(k.GT.imax)) THEN
          !WRITE(6,*) k," iterations"
          EXIT
       END IF
       
       x0=x
       rj=rj1
       rhoj=rhoj1
       
    END DO

    DEALLOCATE(x0)
    DEALLOCATE(rj)
    DEALLOCATE(s)
    DEALLOCATE(rj1)
    DEALLOCATE(zj,yj)
    DEALLOCATE(p)
    DEALLOCATE(q,u,v)
    DEALLOCATE(vtmp1,vtmp2)

    RETURN
  END FUNCTION cgs


  FUNCTION bicgstab(A,IA,JA,b,eps,itermax,BILUT,ibilut,jbilut,debug) RESULT(x)
    USE BLAS95
    IMPLICIT NONE
    !---------------------------------------------------
    ! Biconjugate Gradient Stabilized reel pour resolution du
    ! systeme lineaire reel Ax=b non symetrique
    ! Christophe Besse 2010
    ! Variables d'entree   : matrice A
    !                        2nd membre b
    !                        precision eps (OPTION)
    !                        nb iter maxi itermax (OPTION)
    !                        debug (OPTION)
    !
    ! Variables de sorties : resultat x
    !---------------------------------------------------
    INTEGER ,INTENT(in), DIMENSION(:) :: IA,JA
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: b
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: A
    REAL (kind=8), OPTIONAL, INTENT(in) :: eps
    INTEGER, OPTIONAL, INTENT(in) :: itermax
    INTEGER, OPTIONAL, INTENT(in) :: debug
    REAL (kind=8) , OPTIONAL, INTENT(in), DIMENSION(:) :: BILUT
    INTEGER , OPTIONAL, INTENT(in), DIMENSION(:) :: IBILUT,JBILUT
    REAL (kind=8), DIMENSION(SIZE(b)) :: x

    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: x0,s0,rj,rj1,p,q,vtmp,vtmq
    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: sj,pj,yj,qj
    REAL (kind=8) :: alpha,beta,omega
    REAL (kind=8) :: prec
    INTEGER :: n,k

    n=SIZE(b)
    ALLOCATE(x0(n),sj(n),pj(n),yj(n))
    ALLOCATE(rj(n))
    ALLOCATE(rj1(n))
    ALLOCATE(p(n))
    ALLOCATE(q(n),qj(n))
    ALLOCATE(vtmp(n))
    ALLOCATE(vtmq(n))

    IF (PRESENT(EPS)) THEN
       prec=eps
    ELSE
       prec = 1.e-6
    END IF

    k=0
    x0(:)=0.d0
    CALL MKL_DCSRSYMV('U', N, A, IA, JA, x0,rj)
    rj=b-rj    !rj=b-A*x0
    CALL RANDOM_NUMBER(sj)
    p=rj

    IF (PRESENT(debug)) THEN
       WRITE(6,*) "Debug"
    END IF

    DO
       IF (PRESENT(BILUT)) THEN
          call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, p,yj)    !yj=lsol(M1,p) 
          call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, yj,pj)   !pj=usol(M2,yj)
       ELSE
          pj=p
       END IF
       CALL MKL_DCSRGEMV('N', N, A, IA, JA, pj,vtmp)       !vtmp=A*pj
       alpha=dot(rj,sj)/dot(vtmp,sj)
       q=rj-alpha*vtmp
       
       !check norm of q; if small enough, set x=x+alpha*pj
       IF (nrm2(q).LT.prec) THEN
          !WRITE(6,*) k," iterations"
          x=x0+alpha*pj
          x0=x
          EXIT
       END IF

       IF (PRESENT(BILUT)) THEN
          call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, q,yj)    !yj=lsol(M1,q) 
          call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, yj,qj)   !qj=usol(M2,yj)
       ELSE
          qj=q
       END IF
       CALL MKL_DCSRGEMV('N', N, A, IA, JA, qj,vtmq)       !vtmq=A*qj

       omega=dot(vtmq,q)/dot(vtmq,vtmq)
       x=x0+alpha*pj+omega*qj
       rj1=q-omega*vtmq
       beta=(dot(rj1,sj)/dot(rj,sj))*(alpha/omega)
       p=rj1+beta*(p-omega*vtmp)


       IF (PRESENT(itermax)) THEN
          k=k+1
          IF ((testcv(x0,x,prec,debug)).OR.(k.GT.itermax)) THEN
             WRITE(6,*) k," iterations"
             EXIT
          END IF
       ELSE
          IF (testcv(x0,x,prec,debug)) EXIT
       END IF

       x0=x
       rj=rj1

    END DO

    DEALLOCATE(x0,sj,yj,pj)
    DEALLOCATE(rj)
    DEALLOCATE(rj1)
    DEALLOCATE(p)
    DEALLOCATE(q,qj)
    DEALLOCATE(vtmp)
    DEALLOCATE(vtmq)


    RETURN
  END FUNCTION bicgstab

  FUNCTION gmres(A,IA,JA,b,restart,prec,itermax,BILUT,ibilut,jbilut,debug) RESULT(x)
    USE BLAS95
    IMPLICIT NONE
    !---------------------------------------------------
    ! GMRES pour resolution du
    ! systeme lineaire reel Ax=b non symetrique
    ! Variables d'entree   : matrice A
    !                        2nd membre b
    !                        precision eps (OPTION)
    !                        nb iter maxi itermax (OPTION)
    !                        debug (OPTION)
    !
    ! Variables de sorties : resultat x
    !---------------------------------------------------
    INTEGER ,INTENT(in), DIMENSION(:) :: IA,JA
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: b
    REAL (kind=8) ,INTENT(in), DIMENSION(:) :: A
    integer (kind=8) :: restart
    REAL (kind=8), OPTIONAL, INTENT(in) :: prec
    INTEGER, OPTIONAL, INTENT(in) :: itermax
    INTEGER, OPTIONAL, INTENT(in) :: debug
    REAL (kind=8) , OPTIONAL, INTENT(in), DIMENSION(:) :: BILUT
    INTEGER , OPTIONAL, INTENT(in), DIMENSION(:) :: IBILUT,JBILUT
    REAL (kind=8), DIMENSION(SIZE(b)) :: x

    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: x0,xmin,r,resvec,h,f,vh1,vh,u1,u2,u,q,stagtest,vrf,y,VecTmp,Vtmp
    REAL (kind=8) ,DIMENSION(:),ALLOCATABLE :: vin
    LOGICAL, DIMENSION(:), allocatable :: ind
    REAL (kind=8) ,DIMENSION(:,:),ALLOCATABLE :: V,QT,RR,W
    REAL (kind=8) :: n2b,tolb,normr,normrmin,phibar,rt,c,s,temp,tol,eps
    INTEGER :: n,outer,inner,flag,imin,jmin,stag,i,j,maxit,k


    n=size(b)
    IF (.NOT.(PRESENT(prec))) THEN
       tol=1.e-6
    ELSE
       tol=prec
    END IF
    IF (.NOT.(PRESENT(itermax))) THEN
       maxit=MIN(n/restart,10)
    ELSE
       maxit=itermax
    END IF

    eps = 2.2204460492503131e-16
    outer = maxit;
    inner = restart;
    n2b=nrm2(b)


    ALLOCATE(r(n),vh1(n),vh(n),resvec(inner*outer+1),x0(n),xmin(n))
    ALLOCATE(u1(n),u2(n),u(n),q(inner+1),stagtest(n),ind(n),vrf(n),y(n),vectmp(size(b)))

    x0=0.d0
    x=x0
    
    ! Set up for the method
    flag = 1;
    xmin = x;                          ! Iterate which has minimal residual so far
    imin = 0;                          ! "Outer" iteration at which xmin was computed
    jmin = 0;                          ! "Inner" iteration at which xmin was computed
    tolb = tol * n2b;                  ! Relative tolerance
    r = b - A * x;                     ! Zero-th residual
    normr = nrm2(r);                   ! Norm of residual
    resvec = 0.d0                      ! Preallocate vector for norm of residuals
    resvec(1) = normr;                 ! resvec(1) = norm(b-A*x0)
    normrmin = normr;                  ! Norm of residual from xmin
    stag = 0;                          ! stagnation of the method
          
    ! loop over "outer" iterations (unless convergence or failure)
    ALLOCATE(V(n,inner+1))             ! Arnoldi vectors
    ALLOCATE(h(inner+1))               ! upper Hessenberg st A*V = V*H ...
    ALLOCATE(QT(inner+1,inner+1))      ! orthogonal factor st QT*H = R
    ALLOCATE(RR(inner,inner))          ! upper triangular factor st H = Q*RR
    ALLOCATE(f(inner))                 ! y = RR\f => x = x0 + V*y
    ALLOCATE(W(n,inner))               ! W = V*inv(RR)
    ALLOCATE(Vtmp(n),Vin(n))

    DO i = 1 , outer
       V=0.d0;h=0.d0;QT=0.d0;RR=0.d0;f=0.d0;W=0.d0;

       j=0;                           ! "inner" iteration counter

       IF (PRESENT(BILUT)) THEN
          call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, r,vh1)    !vh1=lsol(M1,r) 
          call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut,vh1,vh)    !vh=usol(M2,vh1)          
       ELSE
          vh=r
       END IF
       h(1)=nrm2(vh)
       v(:,1)=vh/h(1)
       QT(1,1)=1
       phibar=h(1)

       DO j=1,inner

          Vtmp(:)=V(:,j)
          CALL MKL_DCSRGEMV('N', N, A, IA, JA, vtmp,u2)       !u2=A*Vtmp

          IF (PRESENT(BILUT)) THEN
             call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, u2,u1)    !u1=lsol(M1,u2)
             call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, u1,u)    !u=usol(M2,u1) 
          ELSE
             u=u2
          END IF
          

          DO k=1,j
             h(k)=dot(u,V(:,k))
             call axpy(V(:,k),u,-h(k)) !u=u-h(k)*V(:,k)
          END DO
          
          h(j+1)=nrm2(u)
          V(:,j+1)=u/h(j+1)
          call gemv(QT(1:j,1:j),h(1:j),RR(1:j,j)) !RR(1:j,j)=MATMUL(QT(1:j,1:j),h(1:j))
          rt=RR(j,j)

          ! find cos(theta) and sin(theta) of Givens rotation
          IF (h(j+1).EQ.0.d0) THEN
             c = 1.0d0;                      ! theta = 0
             s = 0.0d0;
          ELSEIF (ABS(h(j+1)) > ABS(rt)) then
             temp = rt / h(j+1);
             s = 1.0d0 / SQRT(1.0d0 + ABS(temp)**2); ! pi/4 < theta < 3pi/4
             c = - temp * s;
          ELSE
             temp = h(j+1) / rt;
             c = 1.0d0 / SQRT(1.0d0 + ABS(temp)**2); ! -pi/4 <= theta < 0 < theta <= pi/4
             s = - temp * c;
          END IF
          RR(j,j) = c * rt - s * h(j+1);
          !   zero = s * rt + c * h(j+1);

          q(1:j) = QT(j,1:j);
          QT(j,1:j) = c * q(1:j);
          QT(j+1,1:j) = s * q(1:j);
          QT(j,j+1) = -s;
          QT(j+1,j+1) = c;
          f(j) = c * phibar;
          phibar = s * phibar;

          IF (j < inner) THEN
             IF (f(j).EQ.0.d0) THEN                 ! stagnation of the method
                stag = 1;
             END IF
             call gemv(W(:,1:j-1),RR(1:j-1,j),Vin)
             W(:,j) = (V(:,j) - Vin)/ RR(j,j);
             !W(:,j) = (V(:,j) - MATMUL(W(:,1:j-1),RR(1:j-1,j)))/ RR(j,j);
             ! Check for stagnation of the method
             IF (stag.EQ.0) THEN
                stagtest = 0.d0
                ind = (x.ne.0.d0);
                IF (.NOT.((i.EQ.1).AND.(j.EQ.1))) THEN
                   WHERE (ind) stagtest=W(:,j)/x(:);
                   where ((.NOT.ind).AND.( W(:,j).NE.0)) stagtest=1.e30
!                   stagtest(ind) = W(ind,j) / x(ind);
!                   stagtest((.NOT.ind).AND.( W(:,j).NE.0)) = 1.e30;
                   IF (ABS(f(j))*maxval(abs(stagtest)) < eps) THEN
                      stag = 1;
                   END IF
                END IF
             END IF
             x = x + f(j) * W(:,j);       ! form the new inner iterate
          ELSE ! j == inner
             q(1:j)=gauss(RR(1:j,1:j),f(1:j))
             !vrf = MATMUL(V(:,1:j),q(1:j))
             call gemv(V(:,1:j),q(1:j),vrf)
             ! Check for stagnation of the method
             IF (stag.EQ.0) THEN
                stagtest = 0.d0
                ind = (x0.ne.0.d0);
                WHERE (ind) stagtest=vrf/x0
                where ((.not.ind).and.(vrf.ne.0.d0)) stagtest=1.e30
                IF (maxval(abs(stagtest)) < eps) THEN
                   stag = 1;
                END IF
             END IF
             x = x0 + vrf;                ! form the new outer iterate
          END IF


          CALL MKL_DCSRGEMV('N', N, A, IA, JA, x, vectmp)       !vectmp=A*x
          normr=nrm2(b-vectmp)
          resvec((i-1)*inner+j+1) = normr;

          IF (normr.LE.tolb) THEN               ! check for convergence
             IF (j.LT.inner) THEN
                y(1:j)=gauss(RR(1:j,1:j),f(1:j))
                !vrf = MATMUL(V(:,1:j),y(1:j))
                call gemv(V(:,1:j),y(1:j),vrf)
                x = x0 + vrf;     ! more accurate computation of xj
                CALL MKL_DCSRGEMV('N', N, A, IA, JA, x, vectmp)       !vectmp=A*x
                r=b-vectmp
                normr = nrm2(r);
                resvec((i-1)*inner+j+1) = normr;
             END IF
             IF (normr .LE. tolb)  THEN           ! check using more accurate xj
                flag = 0
                exit
             END IF

          END IF

          IF (stag.EQ.1) THEN
             flag=3
             exit
          END IF
          
          IF (normr < normrmin) THEN            ! update minimal norm quantities
             normrmin = normr;
             xmin = x;
             imin = i;
             jmin = j;
          END IF
       END DO

       IF (flag.EQ.1) THEN
          x0=x
          CALL MKL_DCSRGEMV('N', N, A, IA, JA, x0, vectmp)       !vectmp=A*x0
          r=b-vectmp
       ELSE
          exit
       END IF
    END DO

    ! returned solution is that with minimum residual
    IF (flag.EQ.1) THEN
       STOP "GMRES iterated MAXIT times but did not converge."
    ELSEIF (flag.EQ.3) THEN
       STOP "GMRES stagnated (two consecutive iterates were the same)."
    ELSE
       WRITE(6,'(a6,i2,a31,i2,a18,i2,a39,e11.3)') "gmres(",restart,") converged at outer iteration ",i," (inner iteration ",j,") to a solution WITH relative residual ",normr/n2b
    END IF

    DEALLOCATE(r,vh1,vh,resvec,x0,xmin)
    DEALLOCATE(u1,u2,u,q,stagtest,ind,vrf,y)
    DEALLOCATE(V,Vtmp,vectmp, Vin)
    DEALLOCATE(h)
    DEALLOCATE(QT)
    DEALLOCATE(RR) 
    DEALLOCATE(f)  
    DEALLOCATE(W)
  END FUNCTION gmres

  FUNCTION gauss(A,b) result(x)
    USE BLAS95
    IMPLICIT NONE
    REAL (kind=8), INTENT(in), DIMENSION(:,:) :: A
    REAL (kind=8), INTENT(in), dimension(:) :: b
    REAL (kind=8), dimension(size(b)) :: x
    REAL (kind=8), DIMENSION(:,:), allocatable :: AA
    INTEGER :: n,j,k
    REAL (kind=8) :: pivot,tmp
    n=size(b)
    ALLOCATE(AA(n,n+1))
    AA(:,1:n)=A
    AA(:,n+1)=b
    DO j=1,n
       DO k=j+1,n
          pivot=AA(k,j)/AA(j,j)
          call axpy(AA(j,j:n+1),AA(k,j:n+1),-pivot)
          !AA(k,j:n+1)=AA(k,j:n+1)-pivot*AA(j,j:n+1)
       END DO
    END DO
    x(n)=AA(n,n+1)/AA(n,n)
    DO j=n-1,1,-1
       tmp=dot(AA(j,j+1:n),x(j+1:n))
       x(j)=(AA(j,n+1)-tmp)/AA(j,j)
    END DO
    deallocate(AA)
  END FUNCTION gauss

end module methode
