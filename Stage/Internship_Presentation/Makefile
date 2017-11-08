NAME=Real_Root_Isolation_Presentation

pdf:
	pdflatex ${NAME}.tex
	pdflatex ${NAME}.tex
	pdflatex ${NAME}.tex

clean:
	rm -rf $(NAME).ps $(NAME).dvi $(NAME).log $(NAME).vrb
	rm -rf $(NAME).blg  $(NAME).nav $(NAME).snm $(NAME).bbl
	rm -rf $(NAME).aux $(NAME).fot $(NAME).out $(NAME).toc
	rm -rf *~

cleanall:
	rm -rf $(NAME).ps $(NAME).dvi $(NAME).log $(NAME).vrb
	rm -rf $(NAME).blg  $(NAME).nav $(NAME).snm $(NAME).bbl
	rm -rf $(NAME).aux $(NAME).fot $(NAME).out $(NAME).toc
	rm -rf *~
	rm -rf $(NAME).pdf
