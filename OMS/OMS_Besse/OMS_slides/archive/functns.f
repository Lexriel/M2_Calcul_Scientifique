c-----------------------------------------------------------------------
c     contains the functions needed for defining the PDE poroblems. 
c
c     first for the scalar 5-point and 7-point PDE 
c-----------------------------------------------------------------------
      function afun (x,y,z)
      real*8 afun, x,y, z 
      afun = -1.0D0
      return 
      end
      
      function bfun (x,y,z)
      real*8 bfun, x,y, z 
      bfun = -1.0D0
      return 
      end
      
      function cfun (x,y,z)
      real*8 cfun, x,y, z 
      cfun = -1.0D0
      return 
      end
      
      function dfun (x,y,z)
      real*8 dfun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      dfun = gammax*exp(x*y)
      return 
      end
      
      function efun (x,y,z)
      real*8 efun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      efun = gammay*exp(-x*y) 
      return 
      end
      
      function ffun (x,y,z)
      real*8 ffun, x,y, z 
      ffun = 0.0D0
      return 
      end
      
      function gfun (x,y,z)
      real*8 gfun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      gfun = alpha 
      return 
      end
      
      function hfun (x,y,z)
      real*8 hfun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      hfun = alpha * sin(gammax*x+gammay*y-z)
      return 
      end
      

      function betfun(side, x, y, z)
      real*8 betfun, x, y, z
      character*2 side
      betfun = 1.0
      return
      end

      function gamfun(side, x, y, z)
      real*8 gamfun, x, y, z
      character*2 side
      if (side.eq.'x2') then
         gamfun = 5.0
      else if (side.eq.'y1') then
         gamfun = 2.0
      else if (side.eq.'y2') then
         gamfun = 7.0
      else
         gamfun = 0.0
      endif
      return
      end
c-----------------------------------------------------------------------
c     functions for the block PDE's 
c-----------------------------------------------------------------------
      subroutine afunbl (nfree,x,y,z,coeff)
      return
      end
c     
      subroutine bfunbl (nfree,x,y,z,coeff)
      return 
      end
      
      subroutine cfunbl (nfree,x,y,z,coeff)
c     
      return 
      end
      
      subroutine dfunbl (nfree,x,y,z,coeff)
      
      return
      end
c     
      subroutine efunbl (nfree,x,y,z,coeff)
      return 
      end
c     
      subroutine ffunbl (nfree,x,y,z,coeff)
      return 
      end
c     
      subroutine gfunbl (nfree,x,y,z,coeff)
      return 
      end
      
c$$$      function afun (x,y,z)
c$$$      real*8 afun, x,y,z 
c$$$      afun = -1.0d0
c$$$      return 
c$$$      end
c$$$
c$$$      function bfun (x,y,z)
c$$$      real*8 bfun, x,y,z 
c$$$      bfun = -1.0d0
c$$$      return 
c$$$      end
c$$$
c$$$      function cfun (x,y,z)
c$$$      real*8 cfun, x,y,z 
c$$$      cfun = -1.0d0
c$$$      return 
c$$$      end
c$$$
c$$$      function dfun (x,y,z)
c$$$      real*8 dfun, x,y,z 
c$$$      data gamma /100.0/ 
c$$$c     dfun = gamma * exp( x * y )
c$$$      dfun = 10.d0
c$$$      return 
c$$$      end
c$$$
c$$$      function efun (x,y,z)
c$$$      real*8 efun, x,y,z
c$$$      data gamma /100.0/ 
c$$$c     efun = gamma * exp( (- x) * y ) 
c$$$      efun = 0.d0
c$$$      return 
c$$$      end
c$$$
c$$$      function ffun (x,y,z)
c$$$      real*8 ffun, x,y,z 
c$$$      ffun = 0.0
c$$$      return 
c$$$      end
c$$$
c$$$      function gfun (x,y,z)
c$$$      real*8 gfun, x,y,z 
c$$$      gfun = 0.0 
c$$$      return 
c$$$      end
c$$$
c$$$      function hfun(x, y, z)
c$$$      real*8 hfun, x, y, z
c$$$      hfun = 0.0
c$$$      return
c$$$      end
c$$$
c$$$      function betfun(side, x, y, z)
c$$$      real*8 betfun, x, y, z
c$$$      character*2 side
c$$$      betfun = 1.0
c$$$      return
c$$$      end
c$$$
c$$$      function gamfun(side, x, y, z)
c$$$      real*8 gamfun, x, y, z
c$$$      character*2 side
c$$$      if (side.eq.'x2') then
c$$$         gamfun = 5.0
c$$$      else if (side.eq.'y1') then
c$$$         gamfun = 2.0
c$$$      else if (side.eq.'y2') then
c$$$         gamfun = 7.0
c$$$      else
c$$$         gamfun = 0.0
c$$$      endif
c$$$      return
c$$$      end
c$$$
c$$$c-----------------------------------------------------------------------
c$$$c     functions for the block PDE's 
c$$$c-----------------------------------------------------------------------
c$$$      subroutine afunbl (nfree,x,y,z,coeff)
c$$$      real*8 x, y, z, coeff(100) 
c$$$      do 2 j=1, nfree
c$$$         do 1 i=1, nfree
c$$$            coeff((j-1)*nfree+i) = 0.0d0
c$$$ 1       continue
c$$$         coeff((j-1)*nfree+j) = -1.0d0
c$$$ 2    continue
c$$$      return 
c$$$      end
c$$$
c$$$      subroutine bfunbl (nfree,x,y,z,coeff)
c$$$      real*8 x, y, z, coeff(100) 
c$$$      do 2 j=1, nfree
c$$$         do 1 i=1, nfree
c$$$            coeff((j-1)*nfree+i) = 0.0d0
c$$$ 1       continue
c$$$         coeff((j-1)*nfree+j) = -1.0d0
c$$$ 2    continue
c$$$      return 
c$$$      end
c$$$
c$$$      subroutine cfunbl (nfree,x,y,z,coeff)
c$$$      real*8 x, y, z, coeff(100) 
c$$$      do 2 j=1, nfree
c$$$         do 1 i=1, nfree
c$$$            coeff((j-1)*nfree+i) = 0.0d0
c$$$ 1       continue
c$$$         coeff((j-1)*nfree+j) = -1.0d0
c$$$ 2    continue
c$$$      return 
c$$$      end
c$$$
c$$$      subroutine dfunbl (nfree,x,y,z,coeff)
c$$$      real*8 x, y, z, coeff(100) 
c$$$      do 2 j=1, nfree
c$$$         do 1 i=1, nfree
c$$$            coeff((j-1)*nfree+i) = 0.0d0
c$$$ 1       continue
c$$$ 2    continue
c$$$      return 
c$$$      end
c$$$
c$$$      subroutine efunbl (nfree,x,y,z,coeff)
c$$$      real*8 x, y, z, coeff(100) 
c$$$      do 2 j=1, nfree
c$$$         do 1 i=1, nfree
c$$$            coeff((j-1)*nfree+i) = 0.0d0
c$$$ 1       continue
c$$$ 2    continue
c$$$      return 
c$$$      end
c$$$
c$$$      subroutine ffunbl (nfree,x,y,z,coeff)
c$$$      real*8 x, y, z, coeff(100) 
c$$$      do 2 j=1, nfree
c$$$         do 1 i=1, nfree
c$$$            coeff((j-1)*nfree+i) = 0.0d0
c$$$ 1       continue
c$$$ 2    continue
c$$$      return 
c$$$      end
c$$$
c$$$      subroutine gfunbl (nfree,x,y,z,coeff)
c$$$      real*8 x, y, z, coeff(100) 
c$$$      do 2 j=1, nfree
c$$$         do 1 i=1, nfree
c$$$            coeff((j-1)*nfree+i) = 0.0d0
c$$$ 1       continue
c$$$ 2    continue
c$$$      return 
c$$$      end
c$$$c-----------------------------------------------------------------------
c$$$c     The material property function xyk for the 
c$$$c     finite element problem 
c$$$c-----------------------------------------------------------------------
c$$$      subroutine xyk(nel,xyke,x,y,ijk,node)
c$$$      implicit real*8 (a-h,o-z)
c$$$      dimension xyke(2,2), x(*), y(*), ijk(node,*)
c$$$c     
c$$$c     this is the identity matrix.
c$$$c     
c$$$      xyke(1,1) = 1.0d0
c$$$      xyke(2,2) = 1.0d0
c$$$      xyke(1,2) = 0.0d0
c$$$      xyke(2,1) = 0.0d0
c$$$
c$$$      return
c$$$      end
