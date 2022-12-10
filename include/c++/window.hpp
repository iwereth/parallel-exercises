#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <string>
#include <vector>
#include <array>

class ImageData{
public:
	SDL_Surface* image_surface; //I settled down at using SDL_Surface
								//will always be ARGB8888
								//convenient solution, no direct way to work with pixels buffer
	int x_pos, y_pos, r_height, r_width; //rendered data
	int height, width; //actual height/width
	bool source_sdl;

	ImageData(SDL_Surface* image_surface, int x_pos, int y_pos,
		int r_height, int r_width, bool source_sdl);
};

class Window{
	SDL_Window* main_window;
	SDL_Surface* main_surface;
	bool is_initialized;
	const char* err_string;
	std::vector<ImageData*> images;
	SDL_PixelFormat form;

	void error(const std::string& str);

public:
	Window(int height, int width, std::string title = "CUDA Test");

	ImageData& add_image(const std::string&, int x, int y, int h, int w);
	ImageData& add_image(const unsigned char* bytes, int bh, int bw, int x , int y, int h , int w);

	std::vector<ImageData*>& get_images(void); 
	void show(void);

	~Window();
};