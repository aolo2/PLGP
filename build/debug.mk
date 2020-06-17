include build/build.mk

BUILD_PATH = build/debug
CFLAGS += -g # -fsanitize=address

all:
	@mkdir -p $(BUILD_PATH)
	@/usr/bin/time -f "[TIME] [%E] Built executable $(BUILD_PATH)/$(APP_NAME)" $(CC) $(CFLAGS) $(CFILE) -o $(BUILD_PATH)/$(APP_NAME) $(LDFLAGS)
