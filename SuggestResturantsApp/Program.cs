// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using RestaurantRecommender;

Console.WriteLine("Add started");



























static void Trainig()
{
    MLContext mlContext  = new MLContext();
    var trainigDataFile = "Data\\trainingData.tsv";

    DataPreparer.PreprocessData(trainigDataFile);

    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(trainigDataFile, hasHeader: true);
}