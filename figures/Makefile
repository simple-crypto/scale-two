PNG_LRPRF=lrprf
$(PNG_LRPRF): fig-lr-prf.tex lr-prf.tex
	latexmk -pdf fig-lr-prf.tex
	pdftoppm fig-lr-prf.pdf -png -singlefile $(PNG_LRPRF)
	latexmk -C

