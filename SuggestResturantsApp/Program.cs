// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using RestaurantRecommender;

Console.WriteLine("Add started");
Trainig();


























static void Trainig()
{
    MLContext mlContext = new MLContext();
    var trainingDataFile = "Data\\trainingData.tsv";

    DataPreparer.PreprocessData(trainingDataFile);

    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(trainingDataFile, hasHeader: true);

    var dataPreProcessingPipeline =
        mlContext.Transforms
            .Conversion
            .MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: nameof(ModelInput.UserId))
            .Append(mlContext.Transforms
                .Conversion
                .MapValueToKey(outputColumnName: "RestaurantNameEncoded", inputColumnName: nameof(ModelInput.RestaurantName)));

    var finalOption = new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "UserIdEncoded",
        MatrixRowIndexColumnName = "RestaurantNameEncoded",
        LabelColumnName = "TotalRating",
        NumberOfIterations = 100,
        ApproximationRank = 100,
        Quiet = true
    };
    var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(finalOption);

    var trainerPipeline = dataPreProcessingPipeline.Append(trainer);

    Console.WriteLine("Training Model");

    var model = trainerPipeline.Fit(trainingDataView);
    Console.WriteLine("Which User?");
    var testUserId = Console.ReadLine();

    var predictEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

    var alreadyRatedRestaurant = mlContext.Data.CreateEnumerable<ModelInput>(trainingDataView, false)
        .Where(i=>i.UserId == testUserId).Select(r=>r.RestaurantName).Distinct();

    var allRestaurantsName = trainingDataView.GetColumn<string>("RestaurantName").Distinct()
        .Where(r=>!alreadyRatedRestaurant.Contains(r));


    var scoredRestaurants = allRestaurantsName.Select(restName =>
    {
        var prediction = predictEngine.Predict(new ModelInput()
        {
            RestaurantName = restName,
            UserId = testUserId
        });
        return (RestaurantName: restName , PredictRating: prediction.Score);
    });


    var top10Restaurants = scoredRestaurants
        .OrderByDescending(s => s.PredictRating).Take(10);

    Console.WriteLine();
    Console.WriteLine("----------------------");

    Console.WriteLine($"top 10 restaurants for {testUserId}");
    Console.WriteLine("-------");
    foreach (var top10Restaurant in top10Restaurants)
    {
        Console.WriteLine($"Predicted Rating [{top10Restaurant.PredictRating:#.0}] for restaurant {top10Restaurant.RestaurantName}");
    }

}