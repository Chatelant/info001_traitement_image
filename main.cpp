#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

std::vector<double> histogramme(Mat image)
{
    std::vector<double> histogramme(256, 0.0);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int greyLevel = (int)image.at<uchar>(i, j);
            histogramme[greyLevel]++;
        }
    }

    for (int i = 0; i < histogramme.size(); i++)
    {
        histogramme[i] = histogramme[i] / (image.rows * image.cols);
    }

    return histogramme;
}

std::vector<double> histogramme_cumule(const std::vector<double> &h_I)
{
    std::vector<double> histogrammeCumule(256, 0.0);

    for (int i = 0; i < h_I.size(); i++)
    {
        if (i == 0)
        {
            histogrammeCumule[i] = h_I[i];
        }
        else
        {
            histogrammeCumule[i] = h_I[i] + histogrammeCumule[i - 1];
        }
    }

    return histogrammeCumule;
}

cv::Mat afficheHistogrammes(const std::vector<double> &h_I,
                            const std::vector<double> &H_I)
{
    cv::Mat image(256, 512, CV_8UC1, 255.0);

    // Affichage de l'histogramme
    auto maxValue_h_I = std::max_element(h_I.begin(), h_I.end());

    for (int i = 0; i < h_I.size(); i++)
    {
        auto newValue = (((h_I[i]) * image.rows) / *maxValue_h_I);
        for (int j = 0; j < newValue; j++)
        {
            image.at<uchar>(image.rows - 1 - j, i) = 0;
        }
    }

    // Affichage de l'histogramme cumulé
    auto maxValue_H_I = std::max_element(H_I.begin(), H_I.end());

    for (int i = 0; i < H_I.size(); i++)
    {
        auto newValue = (((H_I[i]) * image.rows) / *maxValue_H_I);
        for (int j = 0; j < newValue; j++)
        {
            image.at<uchar>(image.rows - 1 - j, h_I.size() + i) = 0;
        }
    }

    return image;
}

cv::Mat equalization(Mat image,
                     std::vector<double> &h_I,
                     std::vector<double> &H_I)
{
    // Parcours tout les pixels de l'image
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int greyLevel = (int)image.at<uchar>(i, j);
            image.at<uchar>(i, j) = 255.0 * (double)H_I[greyLevel];
        }
    }
    h_I = histogramme(image);
    H_I = histogramme_cumule(h_I);
    return image;
}

float couleur_la_plus_proche(float pixel)
{
    if (pixel < 255.0 / 2)
    {
        return 0.0;
    }
    else
    {
        return 255.0;
    }
}

void tramage_floyd_steinberg(cv::Mat input, cv::Mat output)
{
    input.convertTo(input, CV_32FC3);
    std::vector<cv::Mat> rgb_channels;
    cv::split(input, rgb_channels);

    for (int c = 0; c < rgb_channels.size(); c++)
    {
        for (int x = 0; x < rgb_channels[c].cols - 1; x++)
        {
            for (int y = 1; y < rgb_channels[c].rows - 1; y++)
            {

                float ancien_pixel = rgb_channels[c].at<float>(y, x);
                float nouveau_pixel = couleur_la_plus_proche(ancien_pixel);
                rgb_channels[c].at<float>(y, x) = nouveau_pixel;
                float erreur_quantification = ancien_pixel - nouveau_pixel;
                rgb_channels[c].at<float>(y + 1, x) = rgb_channels[c].at<float>(y + 1, x) + (7.0 / 16.0) * erreur_quantification;
                rgb_channels[c].at<float>(y - 1, x + 1) = rgb_channels[c].at<float>(y - 1, x + 1) + (3.0 / 16.0) * erreur_quantification;
                rgb_channels[c].at<float>(y, x + 1) = rgb_channels[c].at<float>(y, x + 1) + (5.0 / 16.0) * erreur_quantification;
                rgb_channels[c].at<float>(y + 1, x + 1) = rgb_channels[c].at<float>(y + 1, x + 1) + (1.0 / 16.0) * erreur_quantification;
            }
        }
    }
    cv::merge(rgb_channels, output);
    output.convertTo(output, CV_8UC3);
}

int main(int argc, char *argv[])
{
    if (argv[1] == nullptr)
    {
        std::cout << "\nUsage : ./main <nom-fichier-image>\n"
                  << std::endl;
        exit(1);
    }

    std::string filename = argv[1];
    std::string path = "/home/user/TP1/";
    int old_value = 0;
    int value = 128;

    Mat f = imread(path + filename); // lit l'image  donné en argument
    if (f.channels() > 1)
    {
        cvtColor(f, f, COLOR_RGB2GRAY);
    }

    // Histogrammes
    std::vector<double> hist = histogramme(f);
    std::vector<double> histCumule = histogramme_cumule(hist);

    // Egalisation
    Mat equalizedImg = equalization(f, hist, histCumule);

    cv::Mat displayHistogrammes = afficheHistogrammes(hist, histCumule);
    namedWindow("Histogrammes image egalisee");
    imshow("Histogrammes image egalisee", displayHistogrammes); // l'affiche dans la fenêtre

    // Tramage Floyd Steinberg
    Mat tramedImg(f.rows, f.cols, CV_32FC1, 0.0);
    tramage_floyd_steinberg(f, tramedImg);

    namedWindow("Image de base");
    namedWindow("Image egalisee");
    namedWindow("Image trame");
    imshow("Image de base", f);
    imshow("Image egalisee", equalizedImg);
    imshow("Image trame", tramedImg);
    while (waitKey(50) < 0) // attend une touche
    {                       // Affiche la valeur du slider
        if (value != old_value)
        {
            old_value = value;
            std::cout << "value=" << value << std::endl;
        }
    }
}
