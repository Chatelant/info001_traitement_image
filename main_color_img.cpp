#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

std::vector<double> histogramme(Mat image)
{
    std::vector<double> histogramme(256, 0.0);

    std::vector<cv::Mat> hsv_channels;
    cv::split(image, hsv_channels);
    cv::Mat v_image = hsv_channels[2];

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int lightLevel = (int)v_image.at<uchar>(i, j);
            histogramme[lightLevel]++;
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
    std::vector<cv::Mat> hsv_channels;
    cv::split(image, hsv_channels);

    // Parcours tout les pixels de l'image
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int lightLevel = (int)hsv_channels[2].at<uchar>(i, j);
            hsv_channels[2].at<uchar>(i, j) = 255.0 * (double)H_I[lightLevel];
        }
    }

    cv::merge(hsv_channels, image);

    h_I = histogramme(image);
    H_I = histogramme_cumule(h_I);

    return image;
}

// --- Tramage Floyd --- //
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

// --- Tramage Generique --- //
float distance_color_l2(cv::Vec3f bgr1, cv::Vec3f bgr2)
{
    return sqrt(
        (bgr1[0] - bgr2[0]) * (bgr1[0] - bgr2[0]) +
        (bgr1[1] - bgr2[1]) * (bgr1[1] - bgr2[1]) +
        (bgr1[2] - bgr2[2]) * (bgr1[2] - bgr2[2]));
}

int best_color(cv::Vec3f bgr, std::vector<cv::Vec3f> colors)
{
    int bestColor = 0;
    for (int i = 1; i < colors.size(); i++)
    {
        if (distance_color_l2(bgr, colors[i]) < distance_color_l2(bgr, colors[bestColor]))
            bestColor = i;
    }
    return bestColor;
}

cv::Vec3f error_color(cv::Vec3f bgr1, cv::Vec3f bgr2)
{
    cv::Vec3f error = {0, 0, 0};
    for (int i = 0; i < 3; i++)
    {
        error[i] = bgr1[i] - bgr2[i];
    }
    return error;
}

// cv::Mat tramage_floyd_steinberg(cv::Mat input,
//                                 std::vector<cv::Vec3f> colors)
// {
//     std::vector<cv::Mat> rgb_channels;
//     cv::split(input, rgb_channels);

cv::Mat tramage_floyd_steinberg_generique(cv::Mat input,
                                          std::vector<cv::Vec3f> colors)
{
    // Conversion de input en une matrice de 3 canaux flottants
    cv::Mat fs;
    input.convertTo(fs, CV_32FC3, 1 / 255.0);

    for (int x = 0; x < fs.cols - 1; x++)
    {
        for (int y = 1; y < fs.rows - 1; y++)
        {
            cv::Vec3f c = fs.at<cv::Vec3f>(y, x);
            int i = best_color(c, colors);
            cv::Vec3f e = error_color(c, colors[i]);
            fs.at<cv::Vec3f>(y, x) = colors[i];

            // On propage  aux pixels voisins
            fs.at<cv::Vec3f>(y + 1, x) = fs.at<cv::Vec3f>(y + 1, x) + (7.0 / 16.0) * e;
            fs.at<cv::Vec3f>(y - 1, x + 1) = fs.at<cv::Vec3f>(y - 1, x + 1) + (3.0 / 16.0) * e;
            fs.at<cv::Vec3f>(y, x + 1) = fs.at<cv::Vec3f>(y, x + 1) + (5.0 / 16.0) * e;
            fs.at<cv::Vec3f>(y + 1, x + 1) = fs.at<cv::Vec3f>(y + 1, x + 1) + (1.0 / 16.0) * e;
        }
    }

    // On reconvertit la matrice de 3 canaux flottants en BGR
    cv::Mat output;
    fs.convertTo(output, CV_8UC3, 255.0);
    return output;
}

int main(int argc, char *argv[])
{
    if (argv[1] == nullptr || argv[2] == nullptr)
    {
        std::cout << "\nUsage : ./main_color <nom-fichier-image> <traitement>" << std::endl;
        std::cout << "traitements : none, egal, trame, tram_gen" << std::endl;
        exit(1);
    }

    std::string filename = argv[1];
    std::string path = "/home/user/TP1/";
    std::string traitement = argv[2];

    int old_value = 0;
    int value = 128;

    Mat f = imread(path + filename); // lit l'image  donné en argument

    // ----- None ----- //
    if (traitement == "none")
    {
        namedWindow("Aucun traitement");
        imshow("Aucun traitement", f);
    }

    // ----- Egalisation ----- //
    else if (traitement == "egal")
    {
        // Conversion BGR to HSV
        cv::cvtColor(f, f, COLOR_BGR2HSV);

        // Histogrammes
        std::vector<double> hist = histogramme(f);
        std::vector<double> histCumule = histogramme_cumule(hist);

        // Egalisation
        Mat equalizedImg = equalization(f, hist, histCumule);

        // Conversion HSV to BGR
        cv::cvtColor(equalizedImg, equalizedImg, COLOR_HSV2BGR);

        namedWindow("Image egalise");
        imshow("Image egalise", equalizedImg); // l'affiche dans la fenêtre
        cv::Mat displayHistogrammes = afficheHistogrammes(hist, histCumule);
        namedWindow("Histogrammes");
        imshow("Histogrammes", displayHistogrammes); // l'affiche dans la fenêtre
    }

    // ----- Tramage Floyd Steinberg ----- //
    else if (traitement == "trame")
    {
        Mat tramedImg(f.rows, f.cols, CV_32FC3, 0.0);
        tramage_floyd_steinberg(f, tramedImg);
        namedWindow("TP1 tramage floyd");
        imshow("TP1 tramage floyd", tramedImg); // l'affiche dans la fenêtre
    }

    // ----- Tramage Floyd Steinberg generique ----- //
    else if (traitement == "tram_gen")
    {
        Vec3f blue({1.0, 0.0, 0.0});
        Vec3f green({0.0, 1.0, 0.0});
        Vec3f red({0.0, 0.0, 1.0});
        Vec3f cyan({1.0, 1.0, 0.0});
        Vec3f magenta({1.0, 0.0, 1.0});
        Vec3f yellow({0.0, 1.0, 1.0});
        Vec3f black({0.0, 0.0, 0.0});
        Vec3f white({1.0, 1.0, 1.0});
        std::vector<Vec3f> colors_bgr = {blue, green, red, black, white};
        std::vector<Vec3f> colors_cmjn = {cyan, magenta, yellow, black, white};

        Mat sortie_bgr = tramage_floyd_steinberg_generique(f, colors_bgr);
        Mat sortie_cmjn = tramage_floyd_steinberg_generique(f, colors_cmjn);
        namedWindow("TP1 tramage floyd BGR");          // crée une fenêtre
        namedWindow("TP1 tramage floyd CMJN");         // crée une fenêtre
        imshow("TP1 tramage floyd BGR", sortie_bgr);   // l'affiche dans la fenêtre
        imshow("TP1 tramage floyd CMJN", sortie_cmjn); // l'affiche dans la fenêtre
    }

    while (waitKey(50) < 0) // attend une touche
    {                       // Affiche la valeur du slider
        if (value != old_value)
        {
            old_value = value;
            std::cout << "value=" << value << std::endl;
        }
    }
}