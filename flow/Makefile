default: all

CXX=g++
CFLAGS=-O3 -ftree-vectorize -msse -msse2 -msse3 -mssse3 -fvariable-expansion-in-unroller -march=native -funroll-loops -mfpmath=sse -ffast-math -Wall -m64   -Wextra  -Wunused -Wno-format-y2k  -fno-strict-aliasing

 TARGETDIR = ../build
  TARGET = $(TARGETDIR)/flow.exe
  OBJDIR = ../build/obj
  DEFINES += -D_CRT_SECURE_NO_WARNINGS -DNDEBUG
  INCLUDES += -I.  -I../thitd_party/glfw/include -Ic/VulkanSDK/1.1.82.1/Include
  FORCE_INCLUDE +=
  ALL_CPPFLAGS += $(CPPFLAGS) -MMD -MP $(DEFINES) $(INCLUDES)
  ALL_CFLAGS += $(CFLAGS) $(ALL_CPPFLAGS)   -static  -static-libgcc -static-libstdc++ -s
  ALL_CXXFLAGS += $(CXXFLAGS) $(ALL_CFLAGS)
  ALL_RESFLAGS += $(RESFLAGS) $(DEFINES) $(INCLUDES)
  LIBS += -lglfw3 -lglew32 -lgdi32 -lwinmm -luser32 -lglu32 -lopengl32 -lkernel32  C:/Windows/System32/vulkan-1.dll -static  -static-libgcc -static-libstdc++ -s 
  LDDEPS +=
  ALL_LDFLAGS += $(LDFLAGS) -L/usr/lib64 -L../thitd_party/glfw/lib-mingw-w64 -m64 -s  -static  -static-libgcc -static-libstdc++
  LINKCMD = $(CXX)  $(ALL_LDFLAGS) $(LIBS) $(ALL_CXXFLAGS) -o ${TARGET}
all: 
  $(LINKCMD)

