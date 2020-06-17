include build/build.mk

BUILD_PATH = build/release

CFLAGS += -O2

all:
	@mkdir -p $(BUILD_PATH)
	@cp $(CFILE) $(CUDAFILE)
	@/usr/bin/time -f "[TIME] [%E] Built executable $(BUILD_PATH)/$(APP_NAME)" $(CC) $(CFLAGS) $(CUDAFILE) -o $(BUILD_PATH)/$(APP_NAME) $(LDFLAGS)
	@rm $(CUDAFILE)