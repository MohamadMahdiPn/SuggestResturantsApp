// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using RestaurantRecommender;

Console.WriteLine("Add started");
//Trainig();
TrainigV2();

























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


static void TrainigV2()
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


   
    var crossValMetrics = mlContext.Recommendation()
        .CrossValidate(data: trainingDataView, estimator: trainerPipeline,
        labelColumnName: "TotalRating");

    var averageRMSE = crossValMetrics.Average(m => m.Metrics.RootMeanSquaredError);
    var averageRSquared = crossValMetrics.Average(m => m.Metrics.RSquared);

    Console.WriteLine();
    Console.WriteLine("------------------- Metrics before tuning hyper parameters ----");
    Console.WriteLine($"Cross validated root error : {averageRMSE:#.000}");
    Console.WriteLine($"Cross validated RSquared : {averageRSquared:#.000}");
    Console.WriteLine();
    HyperParameterExploration(mlContext, dataPreProcessingPipeline, trainingDataView);
    var prediction = predictEngine.Predict(new ModelInput()
    {
        RestaurantName = "Restaurant Wu Zhuo Yi",
        UserId = "CLONED",

    });
    Console.WriteLine($"Predicted {prediction.Score:#:0} for Restaurant Wu Zhuo Yi");


}


static void HyperParameterExploration(MLContext mlContext , IEstimator<ITransformer> dataProcessingPipeLine , IDataView trainingDataView)
{
    var results = new List<(double rootMeanSquaredError, double rSquared, int interations, int approximationRank)>();
    for (int interations = 5; interations < 100; interations+=5)
    {
        for (int approximationRank = 50; approximationRank < 250; approximationRank+=50)
        {
            var option = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "UserIdEncoded",
                MatrixRowIndexColumnName = "RestaurantNameEncoded",
                LabelColumnName = "TotalRating",
                NumberOfIterations = interations,
                ApproximationRank = approximationRank,
                Quiet = true
            };
            var currentTrainer = mlContext.Recommendation().Trainers.MatrixFactorization(option);

            var completePipeline = dataProcessingPipeLine.Append(currentTrainer);

            var crossValMetrics = mlContext.Recommendation()
                .CrossValidate(trainingDataView, completePipeline, labelColumnName: "TotalRating");

            results.Add(
                (crossValMetrics.Average(m=>m.Metrics.RootMeanSquaredError),
                    crossValMetrics.Average(m=>m.Metrics.RSquared),
                    interations , approximationRank)
                );
        }
    }


    Console.WriteLine("---  Hyper Parameter and Metrics ----");
    foreach (var valueTuple in results.OrderByDescending(r=>r.rSquared))
    {
        Console.WriteLine($"NUmber of Iterations: {valueTuple.interations}");
        Console.WriteLine($"NUmber of ApproximationRank: {valueTuple.approximationRank}");
        Console.WriteLine($"NUmber of RootMeanSquaredError: {valueTuple.rootMeanSquaredError}");
        Console.WriteLine($"NUmber of rSquared: {valueTuple.rSquared}");
    }

    Console.WriteLine("Done");
}