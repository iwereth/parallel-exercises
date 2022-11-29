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
	int x_pos, y_pos, height, width;

	ImageData(SDL_Surface* image_surface, int x_pos, int y_pos,
		int height, int width);
};

class Window{
	SDL_Window* main_window;
	SDL_Surface* main_surface;
	bool is_initialized;
	const char* err_string;
	std::vector<ImageData*> images;
	SDL_PixelFormat form;

public:
	Window(int height, int width, std::string title = "CUDA Test");

	Window& add_image(const std::string&, int, int, int, int);
	template <std::size_t R, std::size_t C>
	Window& add_image(const std::array<unsigned char, R*C>&, int, int);

	void show();

	~Window();
};