# Compiler
CXX      := nvcc

# Flags
# CXXFLAGS: General compiler flags
# DEPFLAGS: Specifically for dependency generation
# LDFLAGS:  Specifically for the linking phase
CXXFLAGS := -Iinclude -O3
DEPFLAGS = -MMD

TARGET   := bunnyMIP
SRC_DIR  := src
OBJ_DIR  := obj

# Files
SRCS_CPP := $(wildcard $(SRC_DIR)/*.cpp)
SRCS_CU  := $(wildcard $(SRC_DIR)/*.cu)
OBJS     := $(SRCS_CPP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o) \
            $(SRCS_CU:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
DEPS     := $(OBJS:.o=.d)

all: $(TARGET)

# --- LINKING STEP ---
# Notice we use LDFLAGS here, NOT CXXFLAGS or DEPFLAGS
$(TARGET): $(OBJS)
	$(CXX) $^ -o $@

# --- COMPILATION STEPS ---
# We include DEPFLAGS here so dependencies are only made during .o creation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

-include $(DEPS)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: clean all
