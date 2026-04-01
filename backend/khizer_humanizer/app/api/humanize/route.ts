import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const { text, style } = await request.json()

    if (!text) {
      return NextResponse.json({ error: "Text is required" }, { status: 400 })
    }

    // Forward the request to the Python backend
    const backendUrl = "http://localhost:8000/humanize"

    try {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text, style }),
      })

      const data = await backendResponse.json()
      return NextResponse.json(data)
    } catch (error) {
      console.error("Error connecting to backend:", error)

      // Fallback to simple simulation if backend is not available
      let humanized_text = ""

      // Simple simulation of different styles
      switch (style) {
        case "casual":
          humanized_text = simulateCasualStyle(text)
          break
        case "professional":
          humanized_text = simulateProfessionalStyle(text)
          break
        case "creative":
          humanized_text = simulateCreativeStyle(text)
          break
        default:
          humanized_text = simulateCasualStyle(text)
      }

      return NextResponse.json({
        humanized_text,
        humanness_score: 0.8,
        processing_time: 0.5,
        note: "Using fallback simulation as backend is not available",
      })
    }
  } catch (error) {
    console.error("Error processing request:", error)
    return NextResponse.json({ error: "Failed to process text" }, { status: 500 })
  }
}

// Simple text transformation functions to simulate the Python backend
function simulateCasualStyle(text: string): string {
  // Add some casual markers like contractions, filler words
  const result = text
    .replace(/cannot/g, "can't")
    .replace(/will not/g, "won't")
    .replace(/do not/g, "don't")
    .replace(/\. /g, ". Well, ")
    .replace(/\? /g, "? Hmm, ")

  // Add some casual interjections
  const sentences = result.split(". ")
  if (sentences.length > 2) {
    const randomIndex = Math.floor(Math.random() * (sentences.length - 1)) + 1
    sentences[randomIndex] = "You know what? " + sentences[randomIndex]
  }

  return sentences.join(". ")
}

function simulateProfessionalStyle(text: string): string {
  // Make text more formal and structured
  const result = text
    .replace(/don't/g, "do not")
    .replace(/can't/g, "cannot")
    .replace(/won't/g, "will not")
    .replace(/I think/g, "In my assessment")
    .replace(/I believe/g, "I would posit that")

  // Add some professional phrases
  const sentences = result.split(". ")
  if (sentences.length > 2) {
    const randomIndex = Math.floor(Math.random() * (sentences.length - 1)) + 1
    sentences[randomIndex] = "To elaborate further, " + sentences[randomIndex].toLowerCase()
  }

  return sentences.join(". ")
}

function simulateCreativeStyle(text: string): string {
  // Add more colorful language and metaphors
  const result = text

  // Add some creative elements
  const sentences = result.split(". ")
  if (sentences.length > 2) {
    const randomIndex = Math.floor(Math.random() * (sentences.length - 1)) + 1
    sentences[randomIndex] = "Picture this: " + sentences[randomIndex]

    const anotherIndex = (randomIndex + 2) % sentences.length
    sentences[anotherIndex] = sentences[anotherIndex] + " — like a symphony of ideas"
  }

  return sentences.join(". ")
}
