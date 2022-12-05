#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <iostream>
#include "window.h"

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

int main(){
	Window win(SCREEN_HEIGHT,SCREEN_WIDTH);
	win.add_image("lena_color.tiff", 30,30,256,256);
	win.show();
	SDL_Delay(1000);
	return 0;
}