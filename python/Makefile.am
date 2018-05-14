all-local:
	CFLAGS+="$(PYWRAPPER_EXTRA_CFLAGS) $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) $(JPEG_CFLAGS)" \
	LDFLAGS+="$(PYWRAPPER_EXTRA_LDFLAGS)" \
    CC="$(CXX)" \
    python3 setup.py build

check-local:

clean-local:
	python3 setup.py clean --all
	rm -rf ./nnfc/_ext

install-data-local:

install-exec-local:
	CFLAGS+="$(PYWRAPPER_EXTRA_CFLAGS) $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) $(JPEG_CFLAGS)" \
	LDFLAGS+="$(PYWRAPPER_EXTRA_LDFLAGS)" \
    CC="$(CXX)" \
	python3 setup.py install

uninstall-local:

