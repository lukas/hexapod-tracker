// Adapted from Record3D's MIT-licensed synchronized metadata WebRTC demo.
// The SEI payload contains the exact intrinsics and 6-DoF pose for this frame.

function ebspToRbsp(source: DataView) {
  const output = new Uint8Array(source.byteLength)
  let zeros = 0
  let outputIndex = 0
  for (let index = 0; index < source.byteLength; index += 1) {
    const value = source.getUint8(index)
    if (!(value === 0x03 && zeros >= 2)) output[outputIndex++] = value
    zeros = value === 0 ? zeros + 1 : 0
  }
  return output.subarray(0, outputIndex)
}

function metadataPayload(bytes: Uint8Array) {
  let offset = 0
  while (bytes[offset] === 0xff) offset += 1
  offset += 1 + 16 // final payload-size byte and UUID
  return bytes.subarray(offset, Math.max(offset, bytes.length - 1))
}

self.addEventListener('rtctransform', (rawEvent: Event) => {
  const event = rawEvent as Event & {transformer: {readable: ReadableStream; writable: WritableStream}}
  const transform = new TransformStream({
    transform(encodedFrame: {data: ArrayBuffer; getMetadata: () => {rtpTimestamp?: number}}, controller) {
      const view = new DataView(encodedFrame.data)
      const prefix = new Uint8Array([0x00, 0x00, 0x00, 0x01, 0x06, 0x05])
      let start = -1
      for (let candidate = view.byteLength - prefix.length - 1; candidate > 0; candidate -= 1) {
        let matches = true
        for (let index = 0; index < prefix.length; index += 1) {
          if (view.getUint8(candidate + index) !== prefix[index]) {
            matches = false
            break
          }
        }
        if (matches) {
          start = candidate
          break
        }
      }
      if (start > 0) {
        try {
          const ebsp = new DataView(
            encodedFrame.data,
            start + prefix.length,
            encodedFrame.data.byteLength - start - prefix.length,
          )
          const text = new TextDecoder().decode(metadataPayload(ebspToRbsp(ebsp)))
          self.postMessage({
            frameTimestamp: encodedFrame.getMetadata().rtpTimestamp,
            metadata: JSON.parse(text),
          })
          encodedFrame.data = encodedFrame.data.slice(0, start)
        } catch (error) {
          self.postMessage({error: error instanceof Error ? error.message : String(error)})
        }
      }
      controller.enqueue(encodedFrame)
    },
  })
  event.transformer.readable.pipeThrough(transform).pipeTo(event.transformer.writable)
})

export {}
