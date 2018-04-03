all-local:
	cp -dR --preserve=timestamps $(srcdir)/libjpeg-turbo ./libjpeg-turbo.build
	cd libjpeg-turbo.build && autoreconf -fiv && sh configure --prefix=$(shell pwd)/libjpeg-turbo.compiled && $(MAKE) install CFLAGS+=-fPIC

check-local:

clean-local:
	rm -rf libjpeg-turbo.build
	rm -rf libjpeg-turbo.compiled

install-data-local:

install-exec-local:

uninstall-local:
