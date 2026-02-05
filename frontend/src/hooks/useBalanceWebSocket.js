import { useState, useEffect, useRef, useCallback } from 'react'

const WS_RECONNECT_DELAY = 3000

export default function useBalanceWebSocket(enabled = true) {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  const connect = useCallback(() => {
    if (!enabled) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/api/portfolio/ws/balance`

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current)
          reconnectTimer.current = null
        }
      }

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data)
          setData(parsed)
        } catch (e) {
          console.error('WebSocket parse error:', e)
        }
      }

      ws.onclose = () => {
        setConnected(false)
        wsRef.current = null
        // 자동 재연결
        if (enabled) {
          reconnectTimer.current = setTimeout(connect, WS_RECONNECT_DELAY)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    } catch (e) {
      console.error('WebSocket connection error:', e)
    }
  }, [enabled])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [connect])

  return { data, connected }
}
