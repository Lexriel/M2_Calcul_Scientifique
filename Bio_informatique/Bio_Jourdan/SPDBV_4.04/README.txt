
             DeepView / Swiss-PdbViewer
             ==========================

            http://spdbv.vital-it.ch/
            http://www.expasy.org/spdbv/

-------------------------------------------------------------

DESCRIPTION
===========

Swiss-PdbViewer is an application that provides a user friendly
interface allowing to analyse several proteins at the same time.
The proteins can be superimposed in order to deduce structural 
alignments and compare their active sites or any other relevant 
parts. Amino acid mutations, H-bonds, angles and distances between
atoms are easy to obtain thanks to the intuitive graphic and menu
interface. 

Moreover, Swiss-PdbViewer is tightly linked to Swiss-Model, an 
automated homology modelling server accessible from ExPASy

Working with these two programs greatly reduces the amount of 
work necessary to generate models, as it is possible to thread
a protein primary sequence onto a 3D template and get an immediate
feedback of how well the threaded protein will be accepted by the
reference structure before submitting a request to build missing 
loops and refine sidechain packing.

NOTE for Win NT:
----------------
SPDBV is writing temporary files while running. Please make
sure, that the installation directory  (e.g. C:\spdbv) and 
all subdirectories have the correct Read/Write permissions
for all users working with this installation.


HISTORY
=======

I started working on Swiss-PdbViewer in 1995 to visualize at home
comparative modeling results of one of the enzymes I cloned during my PhD
(soybean malate dehydrogenase). The first version was an adaptation of a
software I wrote to help a friend of mine produce stereograms during
the big "Magic Eye" wave. At that time it was a Mac-only 68K
software and had even some assembly code to speed-up display and
rendering. Later I joined Manuel Peitsch's group at the Glaxo Institute
for Molecular Biology and started to adapt the software to run on
Windows NT and serve as a GUI for SWISSMODEL.
Later, Alexandre Diemand and Torsten Schwede both contributed to
the development: Alexandre ported the GUI to SGI translated QuickDraw3D
calls to openGL and contributed to the development of the scripting language.
Torsten implemented electrostatic computation and hardware stereo support
for the PC version. He also improved the SWISSMODEL server interface and
the backend server allowing to blast and retrieve sequences and structures
from the Swiss-PdbViewer interface and he is now in charge of the
SWISSMODEL server http://swissmodel.expasy.org//SWISS-MODEL.html

I also wish to acknowledge Wilfred van Gunsteren for the GROMOS96
force field parameters and Barry Honig for the grasp file format, and
last but not least, the numerous users who provided feedback, suggestions, 
and bug reports. In particular Holger Scheib, David Mathog, Eric Martz,
Joe Krahn, Simon Andrews, Mike Word and Gale Rohdes, who, besides performing extensive beta testing since version 0.98 made a wonderful tutorial

            http://spdbv.vital-it.ch/TheMolecularLevel/SPVTut/index.html
   
and also maintains a discussion list.

       http://spdbv.vital-it.ch/TheMolecularLevel/SPVTut/text/DiscuSPV.html

Have fun!
Nicolas
