Module GestionErr
Interface ErrAlloc
   Module Procedure ErrAlloc_Tab
   Module Procedure ErrAlloc_Seq
End Interface ! ErrAlloc
Contains
!
Subroutine ErrAlloc_Tab(Ierr,String)
Implicit None
Integer, Dimension(:) :: Ierr
Character (Len=*) String
If(MaxVal(Abs(Ierr)) /= 0 ) Then
   Print*,"Probleme d'allocation memoire pour '",String,"'"
   Stop
Endif
!
Return
End Subroutine ErrAlloc_Tab
!
Subroutine ErrAlloc_Seq(Ierr,String)
Implicit None
Integer :: Ierr
Character (Len=*) String
If ( Ierr /= 0 ) Then
   Print*,"Probleme d'allocation memoire pour '",String,"'"
   Stop
Endif
Return
End Subroutine ErrAlloc_Seq
!
End Module GestionErr
