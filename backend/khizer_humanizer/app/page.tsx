"use client"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { useState } from "react"
import { Loader2, RefreshCw, Copy, CheckCircle, Zap, AlertTriangle } from "lucide-react"

export default function Home() {
  const [inputText, setInputText] = useState("")
  const [outputText, setOutputText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("casual")
  const [temperature, setTemperature] = useState(0.9)
  const [preserveMeaning, setPreserveMeaning] = useState(false)
  const [multiPass, setMultiPass] = useState(true)
  const [copied, setCopied] = useState(false)
  const [processingTime, setProcessingTime] = useState<number | null>(null)
  const [humanScore, setHumanScore] = useState<number | null>(null)
  const [passes, setPasses] = useState<number | null>(null)

  const handleHumanize = async () => {
    if (!inputText.trim()) return

    setIsLoading(true)
    try {
      const response = await fetch("http://localhost:8000/humanize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: inputText,
          style: activeTab,
          temperature: temperature,
          preserve_meaning: preserveMeaning,
          multi_pass: multiPass,
        }),
      })

      const data = await response.json()
      setOutputText(data.humanized_text)
      setProcessingTime(data.processing_time)
      setHumanScore(data.humanness_score)
      setPasses(data.passes ?? 1)
    } catch (error) {
      console.error("Error humanizing text:", error)
      setOutputText("An error occurred while processing your text. Make sure the backend server is running.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(outputText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleRehumanize = () => {
    if (outputText) {
      setInputText(outputText)
      setOutputText("")
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-6 md:p-24 bg-gray-50">
      <div className="w-full max-w-4xl space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">AI Text Humanizer</h1>
          <p className="text-gray-500">
            Transform AI-generated content into natural, human-like text that bypasses detection
          </p>
          <div className="flex items-center justify-center mt-2 text-amber-600 bg-amber-50 p-2 rounded-md">
            <AlertTriangle className="h-4 w-4 mr-2" />
            <span className="text-sm font-medium">Bypass Edition v0.3.0</span>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Input Text</CardTitle>
            <CardDescription>Paste your AI-generated text here to humanize it</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              placeholder="Enter your AI-generated text here..."
              className="min-h-[150px]"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button variant="outline" onClick={() => setInputText("")}>
              Clear
            </Button>
            <Button onClick={handleHumanize} disabled={isLoading || !inputText.trim()}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" /> Humanize Text
                </>
              )}
            </Button>
          </CardFooter>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Style</CardTitle>
              <CardDescription>Choose the writing style for humanization</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="casual" className="w-full" onValueChange={setActiveTab} value={activeTab}>
                <TabsList className="grid grid-cols-3 mb-4">
                  <TabsTrigger value="casual">Casual</TabsTrigger>
                  <TabsTrigger value="professional">Professional</TabsTrigger>
                  <TabsTrigger value="creative">Creative</TabsTrigger>
                </TabsList>
                <TabsContent value="casual">
                  <p className="text-sm text-gray-500">
                    Conversational tone with contractions, simpler vocabulary, and informal expressions.
                  </p>
                </TabsContent>
                <TabsContent value="professional">
                  <p className="text-sm text-gray-500">
                    Formal language with expanded vocabulary, proper grammar, and business-appropriate phrasing.
                  </p>
                </TabsContent>
                <TabsContent value="creative">
                  <p className="text-sm text-gray-500">
                    Expressive language with vivid descriptions, varied sentence structures, and engaging metaphors.
                  </p>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Advanced Settings</CardTitle>
              <CardDescription>Fine-tune the humanization process</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="temperature">Transformation Strength</Label>
                  <span className="text-sm text-gray-500">{temperature.toFixed(1)}</span>
                </div>
                <Slider
                  id="temperature"
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  value={[temperature]}
                  onValueChange={(value) => setTemperature(value[0])}
                />
                <p className="text-xs text-gray-500">
                  Higher values produce more creative and varied transformations that are better at evading detection.
                </p>
              </div>

              <div className="flex items-center space-x-2">
                <Switch id="preserve-meaning" checked={preserveMeaning} onCheckedChange={setPreserveMeaning} />
                <Label htmlFor="preserve-meaning">Preserve Original Meaning</Label>
              </div>
              <p className="text-xs text-gray-500">
                Turn off to allow more aggressive transformations that may alter the original meaning slightly but are
                more effective at bypassing detection.
              </p>

              <div className="flex items-center space-x-2">
                <Switch id="multi-pass" checked={multiPass} onCheckedChange={setMultiPass} />
                <Label htmlFor="multi-pass">Multi-Pass Processing</Label>
              </div>
              <p className="text-xs text-gray-500">
                Applies multiple transformation passes for better results. Recommended for bypassing sophisticated
                detection systems like undetectable.ai.
              </p>
            </CardContent>
          </Card>
        </div>

        {outputText && (
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Humanized Output</CardTitle>
                  <CardDescription>Your text has been transformed to bypass detection</CardDescription>
                </div>
                <div className="flex items-center space-x-2">
                  {humanScore !== null && (
                    <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                      {(humanScore * 100).toFixed(0)}% Human
                    </div>
                  )}
                  {passes !== null && (
                    <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                      {passes} {passes === 1 ? "Pass" : "Passes"}
                    </div>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="p-4 bg-white rounded-md border whitespace-pre-wrap">{outputText}</div>
              {processingTime !== null && (
                <p className="text-xs text-gray-500 mt-2">Processing time: {processingTime.toFixed(2)} seconds</p>
              )}
              <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-md text-sm text-amber-800">
                <p className="font-medium">Tips for bypassing undetectable.ai:</p>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Process text in smaller chunks (1-3 paragraphs) for better results</li>
                  <li>Use the "Re-humanize" button to apply multiple passes</li>
                  <li>Try different writing styles for the same content</li>
                  <li>Manually edit a few words or sentences after processing</li>
                  <li>Combine with other humanization tools for maximum effectiveness</li>
                </ul>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={handleRehumanize}>
                <RefreshCw className="mr-2 h-4 w-4" /> Re-humanize
              </Button>
              <Button variant="outline" onClick={handleCopy}>
                {copied ? (
                  <>
                    <CheckCircle className="mr-2 h-4 w-4" /> Copied
                  </>
                ) : (
                  <>
                    <Copy className="mr-2 h-4 w-4" /> Copy to Clipboard
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        )}
      </div>
    </main>
  )
}
