import { createContext, useContext, useState, useEffect } from 'react'
import { authLogin, authLogout, authCheck, authRegister, setAccessToken, clearAccessToken } from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // 초기 로그인 상태 확인
    checkAuthStatus()
  }, [])

  const checkAuthStatus = async () => {
    try {
      // 서버에 인증 상태 확인 (Cookie 기반)
      const response = await authCheck()

      if (response.authenticated) {
        setIsLoggedIn(true)
        setUser({
          username: response.username,
          nickname: response.nickname,
          accountNo: response.account_no,
        })
      } else {
        setIsLoggedIn(false)
        setUser(null)
      }
    } catch (error) {
      console.error('Auth check failed:', error)
      setIsLoggedIn(false)
      setUser(null)
    } finally {
      setIsLoading(false)
    }
  }

  const register = async (data) => {
    try {
      const response = await authRegister(data)

      if (response.success) {
        return { success: true, message: response.message }
      } else {
        return { success: false, message: response.message }
      }
    } catch (error) {
      console.error('Register failed:', error)
      const message = error.response?.data?.detail || error.message || '회원가입에 실패했습니다'
      return { success: false, message }
    }
  }

  const login = async (credentials) => {
    try {
      const response = await authLogin(credentials)

      if (response.success) {
        // 메모리에 토큰 저장 (API 요청 시 사용)
        if (response.access_token) {
          setAccessToken(response.access_token)
        }

        setIsLoggedIn(true)
        setUser({
          username: response.username,
          nickname: response.nickname,
          accountNo: response.account_no,
        })

        return { success: true, message: response.message }
      } else {
        return { success: false, message: response.message }
      }
    } catch (error) {
      console.error('Login failed:', error)
      const message = error.response?.data?.detail || error.message || '로그인에 실패했습니다'
      return { success: false, message }
    }
  }

  const logout = async () => {
    try {
      await authLogout()
    } catch (error) {
      console.error('Logout API failed:', error)
    } finally {
      // API 실패해도 로컬 상태는 정리
      clearAccessToken()
      setIsLoggedIn(false)
      setUser(null)
    }
  }

  const value = {
    isLoggedIn,
    user,
    isLoading,
    register,
    login,
    logout,
    checkAuthStatus,
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext
