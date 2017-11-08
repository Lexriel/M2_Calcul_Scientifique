############################
#  Building 
############################

all: so 
	-mv *.linkinfo bin/
	-mv libcumodp.so bin/
	
so:
	cd ../src/ && make && mv libcumodp.so ../tests/ &&cd ../tests/

clean:
	rm -f bin/*
	rm -f *.o *~ *.linkinfo *.out *.ptx *.log *.prof *.obj
	rm -f lib$(CUMODP).*
	rm -f main exe
	rm -f $(TEST_OBJECTS)
	rm -f *.log
	rm -f log/*_console.log
	cd ../src/ && make clean && cd ../tests/
	rm -f src.cu src.c

####################END OF THE FILE##################################
