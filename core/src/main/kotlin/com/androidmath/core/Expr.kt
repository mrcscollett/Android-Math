package com.androidmath.core

import kotlin.math.*

/**
 * A compiled real-valued expression in x, y and the constant C.
 *
 * The grammar is the subset of Python/SymPy syntax the engine emits for
 * plotting: numbers, x, y, C, pi, E, + - * / **, parentheses and the
 * functions listed in [FUNCTIONS].
 */
fun interface Expr {
    fun eval(x: Double, y: Double, c: Double): Double

    companion object {
        /** Returns null when the text uses anything the evaluator does not support. */
        fun parseOrNull(text: String?): Expr? {
            if (text.isNullOrBlank()) return null
            return try {
                Parser(text).parse()
            } catch (e: IllegalArgumentException) {
                null
            }
        }
    }
}

private val FUNCTIONS: Map<String, (Double) -> Double> = mapOf(
    "sin" to ::sin, "cos" to ::cos, "tan" to ::tan,
    "sec" to { v -> 1 / cos(v) }, "csc" to { v -> 1 / sin(v) }, "cot" to { v -> cos(v) / sin(v) },
    "asin" to ::asin, "acos" to ::acos, "atan" to ::atan,
    "sinh" to ::sinh, "cosh" to ::cosh, "tanh" to ::tanh,
    "exp" to ::exp, "log" to ::ln, "sqrt" to ::sqrt, "Abs" to ::abs,
)

private class Parser(private val src: String) {
    private var pos = 0

    fun parse(): Expr {
        val e = sum()
        skipSpace()
        require(pos == src.length) { "unexpected input at $pos" }
        return e
    }

    private fun skipSpace() {
        while (pos < src.length && src[pos].isWhitespace()) pos++
    }

    private fun peek(): Char {
        skipSpace()
        return if (pos < src.length) src[pos] else '\u0000'
    }

    private fun sum(): Expr {
        var left = product()
        while (true) {
            val op = peek()
            if (op != '+' && op != '-') return left
            pos++
            val a = left
            val b = product()
            left = if (op == '+') Expr { x, y, c -> a.eval(x, y, c) + b.eval(x, y, c) }
            else Expr { x, y, c -> a.eval(x, y, c) - b.eval(x, y, c) }
        }
    }

    private fun product(): Expr {
        var left = unary()
        while (true) {
            val op = peek()
            // A single '*' is multiplication; '**' belongs to power().
            val isMul = op == '*' && !(pos + 1 < src.length && src[pos + 1] == '*')
            if (!isMul && op != '/') return left
            pos++
            val a = left
            val b = unary()
            left = if (isMul) Expr { x, y, c -> a.eval(x, y, c) * b.eval(x, y, c) }
            else Expr { x, y, c -> a.eval(x, y, c) / b.eval(x, y, c) }
        }
    }

    private fun unary(): Expr {
        return when (peek()) {
            '-' -> {
                pos++
                val a = unary()
                Expr { x, y, c -> -a.eval(x, y, c) }
            }
            '+' -> {
                pos++
                unary()
            }
            else -> power()
        }
    }

    private fun power(): Expr {
        val base = atom()
        skipSpace()
        if (src.startsWith("**", pos)) {
            pos += 2
            val exponent = unary() // right-associative, allows 2**-x
            return Expr { x, y, c -> base.eval(x, y, c).pow(exponent.eval(x, y, c)) }
        }
        return base
    }

    private fun atom(): Expr {
        val ch = peek()
        if (ch == '(') {
            pos++
            val inner = sum()
            require(peek() == ')') { "missing )" }
            pos++
            return inner
        }
        if (ch.isDigit() || ch == '.') {
            val start = pos
            while (pos < src.length && (src[pos].isDigit() || src[pos] == '.')) pos++
            if (pos < src.length && (src[pos] == 'e' || src[pos] == 'E') && pos + 1 < src.length &&
                (src[pos + 1].isDigit() || ((src[pos + 1] == '-' || src[pos + 1] == '+') &&
                    pos + 2 < src.length && src[pos + 2].isDigit()))
            ) {
                pos += 2
                while (pos < src.length && src[pos].isDigit()) pos++
            }
            val value = src.substring(start, pos).toDoubleOrNull()
                ?: throw IllegalArgumentException("bad number")
            return Expr { _, _, _ -> value }
        }
        if (ch.isLetter()) {
            val start = pos
            while (pos < src.length && (src[pos].isLetterOrDigit() || src[pos] == '_')) pos++
            val name = src.substring(start, pos)
            if (peek() == '(') {
                val fn = FUNCTIONS[name] ?: throw IllegalArgumentException("unknown function $name")
                pos++
                val arg = sum()
                require(peek() == ')') { "missing )" }
                pos++
                return Expr { x, y, c -> fn(arg.eval(x, y, c)) }
            }
            return when (name) {
                "x" -> Expr { x, _, _ -> x }
                "y" -> Expr { _, y, _ -> y }
                "C" -> Expr { _, _, c -> c }
                "pi" -> Expr { _, _, _ -> PI }
                "E" -> Expr { _, _, _ -> kotlin.math.E }
                else -> throw IllegalArgumentException("unknown name $name")
            }
        }
        throw IllegalArgumentException("unexpected '$ch' at $pos")
    }
}
