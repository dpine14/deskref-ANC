param([string]$Elf = "firmware/build/debug/deskref.elf")
openocd -f board/stm32f4discovery.cfg -c "program $Elf verify reset exit"