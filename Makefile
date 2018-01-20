
TARGET = mean_shift_demo
CC = nvcc --compiler-options -g

$(TARGET): clean src/$(TARGET).cu 
	@clear
	@echo Building $(TARGET)
	@$(CC) src/$(TARGET).cu -o $(TARGET)
	@echo Done

clean:
	@echo 'Removing executable' $(TARGET)
	@rm -f $(TARGET)