all: debug

debug:
	@echo "[INFO] DEBUG build"
	@+make -f build/debug.mk all --no-print-directory

release:
	@echo "[INFO] RELEASE build"
	@+make -f build/release.mk all --no-print-directory

.PHONY:
	all debug release
