NAME=internship_Alexandre_Temperville

pdf:
	pdflatex ${NAME}.tex
	bibtex ${NAME}
	pdflatex ${NAME}.tex
	pdflatex ${NAME}.tex

clean:
	rm -rf $(NAME).ps $(NAME).dvi $(NAME).log
	rm -rf $(NAME).dvi $(NAME).blg $(NAME).bbl
	rm -rf $(NAME).aux $(NAME).pdf $(NAME).fot $(NAME).out $(NAME).toc
	rm -rf *~
