import { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Activity, TrendingUp, Users, DollarSign, Brain, Target, Sparkles } from "lucide-react";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [edaData, setEdaData] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);

  // Form state
  const [formData, setFormData] = useState({
    age: 30,
    gender: "Male",
    income_level: "Middle",
    marital_status: "Single",
    education_level: "Bachelor's",
    occupation: "Middle",
    purchase_category: "Electronics",
    frequency_of_purchase: 5,
    purchase_channel: "Online",
    brand_loyalty: 3,
    product_rating: 4,
    time_spent_on_research: 2.5,
    social_media_influence: "Medium",
    discount_sensitivity: "Somewhat Sensitive",
    return_rate: 1,
    customer_satisfaction: 7,
    engagement_with_ads: "Medium",
    device_used: "Smartphone",
    payment_method: "Credit Card",
    time_to_decision: 5
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [metricsRes, edaRes, datasetRes] = await Promise.all([
        axios.get(`${API}/model-metrics`),
        axios.get(`${API}/eda-data`),
        axios.get(`${API}/dataset-info`)
      ]);

      setMetrics(metricsRes.data);
      setEdaData(edaRes.data);
      setDatasetInfo(datasetRes.data);
      setLoading(false);
    } catch (error) {
      console.error("Error loading data:", error);
      toast.error("Failed to load dashboard data");
      setLoading(false);
    }
  };

  const handlePredict = async (predictionType) => {
    setPredicting(true);
    setPredictionResult(null);

    try {
      const response = await axios.post(`${API}/predict/${predictionType}`, formData);
      setPredictionResult({
        type: predictionType,
        data: response.data
      });
      toast.success("Prediction completed successfully!");
    } catch (error) {
      console.error("Prediction error:", error);
      toast.error("Prediction failed. Please check your inputs.");
    } finally {
      setPredicting(false);
    }
  };

  const handleDatasetUpload = async () => {
    if (!uploadFile) {
      toast.error("Please select a CSV or XML file");
      return;
    }

    setUploading(true);
    setUploadStatus(null);

    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      const response = await axios.post(`${API}/dataset/upload?retrain=true`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setUploadStatus({ type: "success", message: response.data.message });
      toast.success("Dataset uploaded and models retrained");
      await loadData();
    } catch (error) {
      console.error("Dataset upload error:", error);
      setUploadStatus({ type: "error", message: "Upload failed. Check file format." });
      toast.error("Dataset upload failed");
    } finally {
      setUploading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        <div className="text-center">
          <Brain className="w-16 h-16 text-blue-400 animate-pulse mx-auto mb-4" />
          <p className="text-xl text-white">Loading Intelligence System...</p>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const incomeSpendingEntries = Object.entries(edaData?.income_vs_spending ?? {});
  const incomeSpendingData = incomeSpendingEntries.map(([key, value]) => ({
    income: key,
    spending: value
  }));

  const intentDistributionEntries = Object.entries(edaData?.purchase_intent_distribution ?? {});
  const intentDistributionData = intentDistributionEntries.map(([key, value]) => ({
    name: key,
    value: value
  }));

  const loyaltyStats = edaData?.loyalty_stats ?? null;
  const loyaltyComparisonData = loyaltyStats ? [
    { category: "Members", count: loyaltyStats.members ?? 0, avgSpending: loyaltyStats.member_avg_spending ?? 0 },
    { category: "Non-Members", count: loyaltyStats.non_members ?? 0, avgSpending: loyaltyStats.non_member_avg_spending ?? 0 }
  ] : [];

  const discountImpact = edaData?.discount_impact ?? null;
  const discountImpactData = discountImpact ? [
    { type: "With Discount", amount: discountImpact.with_discount ?? 0 },
    { type: "Without Discount", amount: discountImpact.without_discount ?? 0 }
  ] : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Header */}
      <div className="bg-slate-900/50 backdrop-blur-md border-b border-blue-500/20">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-4">
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-3 rounded-2xl">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Customer Intelligence System</h1>
              <p className="text-blue-200 text-sm">Machine Learning Powered Marketing Analytics</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Metrics Overview */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <Card data-testid="metric-card-accuracy" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-blue-200 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Intent Accuracy
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white">
                  {(metrics.metrics.purchase_intent_accuracy * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>

            <Card data-testid="metric-card-loyalty" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-blue-200 flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  Loyalty Accuracy
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white">
                  {(metrics.metrics.loyalty_accuracy * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>

            <Card data-testid="metric-card-r2" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-blue-200 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Amount R² Score
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white">
                  {(metrics.metrics.amount_r2_score * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>

            <Card data-testid="metric-card-samples" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm text-blue-200 flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Training Samples
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-white">
                  {metrics.dataset_info.total_samples.toLocaleString()}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Main Content Tabs */}
        <Tabs defaultValue="eda" className="space-y-6">
          <TabsList className="bg-slate-800/50 backdrop-blur border border-blue-500/20">
            <TabsTrigger data-testid="tab-eda" value="eda" className="data-[state=active]:bg-blue-600">
              Exploratory Analysis
            </TabsTrigger>
            <TabsTrigger data-testid="tab-predict" value="predict" className="data-[state=active]:bg-blue-600">
              Predictions
            </TabsTrigger>
            <TabsTrigger data-testid="tab-dataset" value="dataset" className="data-[state=active]:bg-blue-600">
              Dataset Overview
            </TabsTrigger>
            <TabsTrigger data-testid="tab-insights" value="insights" className="data-[state=active]:bg-blue-600">
              Business Insights
            </TabsTrigger>
          </TabsList>

          {/* EDA Tab */}
          <TabsContent value="eda" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card data-testid="chart-income-spending" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
                <CardHeader>
                  <CardTitle className="text-white">Income Level vs Average Spending</CardTitle>
                  <CardDescription className="text-blue-200">Purchase patterns across income segments</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={incomeSpendingData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="income" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6' }} />
                      <Bar dataKey="spending" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card data-testid="chart-purchase-intent" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
                <CardHeader>
                  <CardTitle className="text-white">Purchase Intent Distribution</CardTitle>
                  <CardDescription className="text-blue-200">Customer buying behavior patterns</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={intentDistributionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={(entry) => entry.name}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {intentDistributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6' }} />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card data-testid="chart-loyalty-comparison" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
                <CardHeader>
                  <CardTitle className="text-white">Loyalty Program Impact</CardTitle>
                  <CardDescription className="text-blue-200">Members vs Non-Members spending</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={loyaltyComparisonData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="category" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6' }} />
                      <Bar dataKey="avgSpending" fill="#8b5cf6" name="Avg Spending" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card data-testid="chart-discount-impact" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
                <CardHeader>
                  <CardTitle className="text-white">Discount Usage Impact</CardTitle>
                  <CardDescription className="text-blue-200">Effect of discounts on purchase amount</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={discountImpactData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="type" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6' }} />
                      <Bar dataKey="amount" fill="#ec4899" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Predictions Tab - Part 1 */}
          <TabsContent value="predict">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Input Form */}
              <Card data-testid="prediction-form" className="lg:col-span-2 bg-slate-800/50 backdrop-blur border-blue-500/20">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    Customer Profile Input
                  </CardTitle>
                  <CardDescription className="text-blue-200">Enter customer information for AI predictions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-blue-200">Age</Label>
                      <Input
                        type="number"
                        value={formData.age}
                        onChange={(e) => setFormData({ ...formData, age: parseInt(e.target.value) })}
                        className="bg-slate-900 border-blue-500/30 text-white"
                        data-testid="input-age"
                      />
                    </div>

                    <div>
                      <Label className="text-blue-200">Gender</Label>
                      <Select value={formData.gender} onValueChange={(val) => setFormData({ ...formData, gender: val })}>
                        <SelectTrigger data-testid="select-gender" className="bg-slate-900 border-blue-500/30 text-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-slate-900 border-blue-500/30">
                          <SelectItem value="Male">Male</SelectItem>
                          <SelectItem value="Female">Female</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label className="text-blue-200">Income Level</Label>
                      <Select value={formData.income_level} onValueChange={(val) => setFormData({ ...formData, income_level: val })}>
                        <SelectTrigger data-testid="select-income" className="bg-slate-900 border-blue-500/30 text-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-slate-900 border-blue-500/30">
                          <SelectItem value="Low">Low</SelectItem>
                          <SelectItem value="Middle">Middle</SelectItem>
                          <SelectItem value="High">High</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label className="text-blue-200">Purchase Category</Label>
                      <Select value={formData.purchase_category} onValueChange={(val) => setFormData({ ...formData, purchase_category: val })}>
                        <SelectTrigger data-testid="select-category" className="bg-slate-900 border-blue-500/30 text-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-slate-900 border-blue-500/30">
                          <SelectItem value="Electronics">Electronics</SelectItem>
                          <SelectItem value="Clothing">Clothing</SelectItem>
                          <SelectItem value="Home Appliances">Home Appliances</SelectItem>
                          <SelectItem value="Food & Beverages">Food & Beverages</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label className="text-blue-200">Brand Loyalty (1-5)</Label>
                      <Input
                        type="number"
                        min="1"
                        max="5"
                        value={formData.brand_loyalty}
                        onChange={(e) => setFormData({ ...formData, brand_loyalty: parseInt(e.target.value) })}
                        className="bg-slate-900 border-blue-500/30 text-white"
                        data-testid="input-brand-loyalty"
                      />
                    </div>

                    <div>
                      <Label className="text-blue-200">Customer Satisfaction (1-10)</Label>
                      <Input
                        type="number"
                        min="1"
                        max="10"
                        value={formData.customer_satisfaction}
                        onChange={(e) => setFormData({ ...formData, customer_satisfaction: parseInt(e.target.value) })}
                        className="bg-slate-900 border-blue-500/30 text-white"
                        data-testid="input-satisfaction"
                      />
                    </div>
                  </div>

                  <div className="mt-6 flex flex-wrap gap-3">
                    <Button
                      onClick={() => handlePredict('purchase-intent')}
                      disabled={predicting}
                      className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800"
                      data-testid="btn-predict-intent"
                    >
                      Predict Purchase Intent
                    </Button>
                    <Button
                      onClick={() => handlePredict('loyalty')}
                      disabled={predicting}
                      className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800"
                      data-testid="btn-predict-loyalty"
                    >
                      Predict Loyalty
                    </Button>
                    <Button
                      onClick={() => handlePredict('amount')}
                      disabled={predicting}
                      className="bg-gradient-to-r from-pink-600 to-pink-700 hover:from-pink-700 hover:to-pink-800"
                      data-testid="btn-predict-amount"
                    >
                      Predict Spending
                    </Button>
                    <Button
                      onClick={() => handlePredict('satisfaction')}
                      disabled={predicting}
                      className="bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800"
                      data-testid="btn-predict-satisfaction"
                    >
                      Predict Satisfaction
                    </Button>
                    <Button
                      onClick={() => handlePredict('return-rate')}
                      disabled={predicting}
                      className="bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800"
                      data-testid="btn-predict-return"
                    >
                      Predict Return Risk
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Results */}
              <Card data-testid="prediction-result" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
                <CardHeader>
                  <CardTitle className="text-white">Prediction Results</CardTitle>
                  <CardDescription className="text-blue-200">AI-powered insights</CardDescription>
                </CardHeader>
                <CardContent>
                  {predicting ? (
                    <div className="text-center py-12">
                      <Brain className="w-12 h-12 text-blue-400 animate-pulse mx-auto mb-4" />
                      <p className="text-blue-200">Analyzing...</p>
                    </div>
                  ) : predictionResult ? (
                    <div className="space-y-4">
                      {predictionResult.type === 'purchase-intent' && (
                        <>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Predicted Intent:</p>
                            <p className="text-2xl font-bold text-white">{predictionResult.data.prediction}</p>
                          </div>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Confidence:</p>
                            <p className="text-xl font-semibold text-green-400">
                              {(predictionResult.data.confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Probabilities:</p>
                            {Object.entries(predictionResult.data.probability).map(([key, val]) => (
                              <div key={key} className="flex justify-between text-sm text-gray-300 mb-1">
                                <span>{key}:</span>
                                <span>{(val * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </>
                      )}

                      {predictionResult.type === 'loyalty' && (
                        <>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Loyalty Program:</p>
                            <p className="text-2xl font-bold text-white">
                              {predictionResult.data.prediction ? "Will Join" : "Won't Join"}
                            </p>
                          </div>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Probability:</p>
                            <p className="text-xl font-semibold text-green-400">
                              {(predictionResult.data.probability * 100).toFixed(1)}%
                            </p>
                          </div>
                        </>
                      )}

                      {predictionResult.type === 'amount' && (
                        <>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Predicted Spending:</p>
                            <p className="text-2xl font-bold text-white">
                              ${predictionResult.data.predicted_amount.toFixed(2)}
                            </p>
                          </div>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Range:</p>
                            <p className="text-sm text-gray-300">
                              ${predictionResult.data.confidence_interval.lower.toFixed(2)} - 
                              ${predictionResult.data.confidence_interval.upper.toFixed(2)}
                            </p>
                          </div>
                        </>
                      )}

                      {predictionResult.type === 'satisfaction' && (
                        <>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Predicted Satisfaction:</p>
                            <p className="text-2xl font-bold text-white">
                              {predictionResult.data.predicted_satisfaction.toFixed(1)}/10
                            </p>
                          </div>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Confidence:</p>
                            <p className="text-xl font-semibold text-green-400">
                              {(predictionResult.data.confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="pt-2">
                            <div className="bg-slate-700 rounded-full h-2 overflow-hidden">
                              <div 
                                className="bg-gradient-to-r from-emerald-400 to-emerald-600 h-full" 
                                style={{width: `${(predictionResult.data.predicted_satisfaction / 10) * 100}%`}}
                              ></div>
                            </div>
                          </div>
                        </>
                      )}

                      {predictionResult.type === 'return-rate' && (
                        <>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Return Prediction:</p>
                            <p className="text-2xl font-bold text-white">
                              {predictionResult.data.will_return ? "⚠️ Likely to Return" : "✓ Keep Product"}
                            </p>
                          </div>
                          <div>
                            <p className="text-sm text-blue-200 mb-2">Return Probability:</p>
                            <p className={`text-xl font-semibold ${predictionResult.data.will_return ? 'text-red-400' : 'text-green-400'}`}>
                              {(predictionResult.data.probability * 100).toFixed(1)}%
                            </p>
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-400">
                      <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>No predictions yet</p>
                      <p className="text-sm mt-2">Fill the form and click predict</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Dataset Tab */}
          <TabsContent value="dataset">
            <Card data-testid="dataset-info" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
              <CardHeader>
                <CardTitle className="text-white">Dataset Information</CardTitle>
                <CardDescription className="text-blue-200">Overview of the training data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-slate-900/50 p-4 rounded-lg mb-6">
                  <h3 className="text-white font-semibold mb-2">Upload New Dataset</h3>
                  <p className="text-sm text-blue-200 mb-4">Supported formats: CSV, XML. Uploading retrains models.</p>
                  <div className="flex flex-col md:flex-row gap-3 items-start md:items-center">
                    <Input
                      type="file"
                      accept=".csv,.xml"
                      onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
                      className="bg-slate-800/60 text-white border-blue-500/30"
                    />
                    <Button
                      onClick={handleDatasetUpload}
                      disabled={uploading}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      {uploading ? "Uploading..." : "Upload & Retrain"}
                    </Button>
                  </div>
                  {uploadStatus && (
                    <p className={`text-sm mt-3 ${uploadStatus.type === "success" ? "text-green-400" : "text-red-400"}`}>
                      {uploadStatus.message}
                    </p>
                  )}
                </div>

                {datasetInfo && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-slate-900/50 p-4 rounded-lg">
                        <p className="text-sm text-blue-200">Total Rows</p>
                        <p className="text-2xl font-bold text-white">{datasetInfo.shape[0]}</p>
                      </div>
                      <div className="bg-slate-900/50 p-4 rounded-lg">
                        <p className="text-sm text-blue-200">Total Columns</p>
                        <p className="text-2xl font-bold text-white">{datasetInfo.shape[1]}</p>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold text-white mb-3">Sample Data Preview</h3>
                      <div className="overflow-x-auto">
                        <div className="bg-slate-900/50 p-4 rounded-lg">
                          <pre className="text-xs text-gray-300 overflow-x-auto">
                            {JSON.stringify(datasetInfo.sample_data.slice(0, 3), null, 2)}
                          </pre>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Insights Tab */}
          <TabsContent value="insights">
            <Card data-testid="business-insights" className="bg-slate-800/50 backdrop-blur border-blue-500/20">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  Business Insights & Recommendations
                </CardTitle>
                <CardDescription className="text-blue-200">AI-generated strategic insights from customer data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {edaData && (
                    <>
                      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 p-6 rounded-lg border border-blue-500/30">
                        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                          <DollarSign className="w-5 h-5" />
                          Revenue Optimization
                        </h3>
                        <ul className="space-y-2 text-gray-300">
                          <li className="flex items-start gap-2">
                            <span className="text-green-400 mt-1">•</span>
                            <span>
                              Loyalty program members spend <strong className="text-white">
                              ${edaData.loyalty_stats.member_avg_spending.toFixed(2)}</strong> on average, 
                              compared to ${edaData.loyalty_stats.non_member_avg_spending.toFixed(2)} for non-members. 
                              <strong className="text-green-400"> Focus on loyalty conversion campaigns.</strong>
                            </span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="text-green-400 mt-1">•</span>
                            <span>
                              Discounts increase average purchase by 
                              <strong className="text-white"> ${(edaData.discount_impact.with_discount - edaData.discount_impact.without_discount).toFixed(2)}</strong>. 
                              Strategic discount placement can boost revenue.
                            </span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 p-6 rounded-lg border border-purple-500/30">
                        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                          <Users className="w-5 h-5" />
                          Customer Segmentation
                        </h3>
                        <ul className="space-y-2 text-gray-300">
                          <li className="flex items-start gap-2">
                            <span className="text-purple-400 mt-1">•</span>
                            <span>
                              <strong className="text-white">{edaData.loyalty_stats.members}</strong> customers are loyalty members 
                              vs <strong className="text-white">{edaData.loyalty_stats.non_members}</strong> non-members. 
                              Significant opportunity for program growth.
                            </span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="text-purple-400 mt-1">•</span>
                            <span>
                              Top purchase intent: <strong className="text-white">
                              {Object.entries(edaData.purchase_intent_distribution).sort((a, b) => b[1] - a[1])[0][0]}</strong>. 
                              Tailor marketing messages accordingly.
                            </span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-r from-pink-900/50 to-orange-900/50 p-6 rounded-lg border border-pink-500/30">
                        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                          <Target className="w-5 h-5" />
                          Action Items
                        </h3>
                        <ul className="space-y-2 text-gray-300">
                          <li className="flex items-start gap-2">
                            <span className="text-orange-400 mt-1">•</span>
                            <span>Deploy targeted loyalty program enrollment campaigns to high-value customers</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="text-orange-400 mt-1">•</span>
                            <span>Optimize discount strategies based on customer segments and purchase patterns</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="text-orange-400 mt-1">•</span>
                            <span>Personalize product recommendations using purchase intent predictions</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="text-orange-400 mt-1">•</span>
                            <span>Focus on high-income segments for premium product marketing</span>
                          </li>
                        </ul>
                      </div>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
