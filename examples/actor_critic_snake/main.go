package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/mg52/neuralnet"
)

////////////////////////////////////////////////////
// CONFIG
////////////////////////////////////////////////////

// Visualization speed
const renderDelay = 120 * time.Millisecond
const useColor = true

////////////////////////////////////////////////////
// VISUALIZATION HELPERS
////////////////////////////////////////////////////

func clearScreen() {
	fmt.Print("\033[2J\033[H")
}

func color(s, c string) string {
	if !useColor {
		return s
	}
	return c + s + "\033[0m"
}

func draw(env *Snake) {
	clearScreen()

	for y := 0; y < env.H; y++ {
		for x := 0; x < env.W; x++ {
			isHead := false
			isBody := false

			if env.Body[0][0] == x && env.Body[0][1] == y {
				isHead = true
			} else {
				for _, b := range env.Body[1:] {
					if b[0] == x && b[1] == y {
						isBody = true
						break
					}
				}
			}

			switch {
			case isHead:
				fmt.Print(color("O", "\033[32m"))
			case isBody:
				fmt.Print(color("o", "\033[34m"))
			case env.FoodX == x && env.FoodY == y:
				fmt.Print(color("X", "\033[31m"))
			default:
				fmt.Print(".")
			}
		}
		fmt.Println()
	}

	fmt.Printf("\nLength: %d\n", env.Length)
	time.Sleep(renderDelay)
}

////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////

func main() {
	// Global deterministic RNG (stable runs)
	globalRng := rand.New(rand.NewSource(1))
	env := NewSnake(11, 11, globalRng)

	fmt.Println("Starting Snake Actor–Critic training")

	stateSize := 11
	actionCount := 3
	hidden := 32
	gamma := 0.99
	episodes := 3000

	actor := neuralnet.NewNetwork(
		stateSize,
		[]int{hidden, hidden},
		actionCount,
		[]string{"ReLU", "ReLU", "Softmax"},
		0.0007,
		"CrossEntropy",
		"Adam",
	)

	critic := neuralnet.NewNetwork(
		stateSize,
		[]int{hidden, hidden},
		1,
		[]string{"ReLU", "ReLU", "Linear"},
		0.001,
		"MSE",
		"Adam",
	)

	////////////////////////////////////////////////////
	// TRAINING LOOP (A2C)
	////////////////////////////////////////////////////

	for ep := 0; ep < episodes; ep++ {
		state := env.Reset()

		var states [][]float64
		var actions []int
		var rewards []float64
		var values []float64

		done := false
		steps := 0
		totalReward := 0.0

		// -------- COLLECT EPISODE --------
		for !done {
			steps++

			policy := actor.Predict(state)
			action := sampleAction(policy, env.rng)
			value := critic.Predict(state)[0]

			nextState, reward, d := env.Step(action)
			done = d

			states = append(states, state)
			actions = append(actions, action)
			rewards = append(rewards, reward)
			values = append(values, value)

			totalReward += reward
			state = nextState

			if steps > env.MaxSteps {
				break
			}
		}

		// -------- RETURNS & ADVANTAGES --------
		n := len(rewards)
		returns := make([]float64, n)
		advantages := make([]float64, n)

		R := 0.0
		for i := n - 1; i >= 0; i-- {
			R = rewards[i] + gamma*R
			returns[i] = R
			advantages[i] = R - values[i]
		}

		mean, std := meanStd(advantages)
		for i := 0; i < n; i++ {
			if std > 1e-8 {
				advantages[i] = (advantages[i] - mean) / std
			} else {
				advantages[i] = 0
			}
		}

		// -------- UPDATE NETWORKS --------
		for i := 0; i < n; i++ {
			s := states[i]
			a := actions[i]
			G := returns[i]
			adv := advantages[i]

			critic.Train(s, []float64{G})

			if adv > 0 {
				pol := actor.Predict(s)
				label := make([]float64, len(pol))
				label[a] = 1.0
				actor.Train(s, label)
			}
		}

		if ep%200 == 0 {
			fmt.Printf(
				"Episode %d: reward=%.2f length=%d steps=%d\n",
				ep, totalReward, env.Length, steps,
			)
		}
	}

	fmt.Println("\nTesting trained Snake agent (visual)...")
	testSnakeVisual(env, actor)
}

////////////////////////////////////////////////////
// VISUAL TEST
////////////////////////////////////////////////////

func testSnakeVisual(env *Snake, actor *neuralnet.Network) {
	state := env.Reset()
	done := false
	steps := 0

	for !done && steps < env.MaxSteps {
		steps++

		draw(env)

		policy := actor.Predict(state)
		action := argMax(policy)

		next, _, end := env.Step(action)
		state = next
		done = end
	}

	draw(env)
	fmt.Println("GAME OVER")
}

////////////////////////////////////////////////////
// STATS
////////////////////////////////////////////////////

func meanStd(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range xs {
		sum += v
	}
	mean := sum / float64(len(xs))

	variance := 0.0
	for _, v := range xs {
		d := v - mean
		variance += d * d
	}
	variance /= float64(len(xs))
	return mean, math.Sqrt(variance)
}

////////////////////////////////////////////////////
// ACTION HELPERS
////////////////////////////////////////////////////

func sampleAction(policy []float64, rng *rand.Rand) int {
	r := rng.Float64()
	sum := 0.0
	for i, p := range policy {
		sum += p
		if r < sum {
			return i
		}
	}
	return len(policy) - 1
}

func argMax(xs []float64) int {
	max := xs[0]
	idx := 0
	for i, v := range xs {
		if v > max {
			max = v
			idx = i
		}
	}
	return idx
}

////////////////////////////////////////////////////
// SNAKE ENVIRONMENT
////////////////////////////////////////////////////

type Snake struct {
	W, H          int
	Body          [][2]int
	Dir           int
	FoodX, FoodY  int
	Length        int
	StepsSinceEat int
	MaxSteps      int

	rng *rand.Rand
}

func NewSnake(w, h int, globalRng *rand.Rand) *Snake {
	return &Snake{
		W:        w,
		H:        h,
		MaxSteps: w * h * 4,
		rng:      rand.New(rand.NewSource(globalRng.Int63())),
	}
}

func (s *Snake) Reset() []float64 {
	s.Body = [][2]int{{s.W / 2, s.H / 2}}
	s.Dir = 0
	s.Length = 1
	s.StepsSinceEat = 0
	s.spawnFood()
	return s.State()
}

func (s *Snake) spawnFood() {
	for {
		x := s.rng.Intn(s.W)
		y := s.rng.Intn(s.H)
		ok := true
		for _, b := range s.Body {
			if b[0] == x && b[1] == y {
				ok = false
				break
			}
		}
		if ok {
			s.FoodX = x
			s.FoodY = y
			return
		}
	}
}

func (s *Snake) danger(x, y int) bool {
	if x < 0 || x >= s.W || y < 0 || y >= s.H {
		return true
	}
	for _, b := range s.Body[1:] {
		if b[0] == x && b[1] == y {
			return true
		}
	}
	return false
}

func (s *Snake) State() []float64 {
	head := s.Body[0]

	// Direction one-hot
	dirUp := boolToFloat(s.Dir == 0)
	dirRight := boolToFloat(s.Dir == 1)
	dirDown := boolToFloat(s.Dir == 2)
	dirLeft := boolToFloat(s.Dir == 3)

	// Relative directions
	var frontDir, leftDir, rightDir int
	switch s.Dir {
	case 0: // up
		frontDir, leftDir, rightDir = 0, 3, 1
	case 1: // right
		frontDir, leftDir, rightDir = 1, 0, 2
	case 2: // down
		frontDir, leftDir, rightDir = 2, 1, 3
	case 3: // left
		frontDir, leftDir, rightDir = 3, 2, 0
	}

	fx, fy := nextPos(head[0], head[1], frontDir)
	lx, ly := nextPos(head[0], head[1], leftDir)
	rx, ry := nextPos(head[0], head[1], rightDir)

	dangerFront := boolToFloat(s.danger(fx, fy))
	dangerLeft := boolToFloat(s.danger(lx, ly))
	dangerRight := boolToFloat(s.danger(rx, ry))

	// Food relative position
	foodUp := boolToFloat(s.FoodY < head[1])
	foodDown := boolToFloat(s.FoodY > head[1])
	foodLeft := boolToFloat(s.FoodX < head[0])
	foodRight := boolToFloat(s.FoodX > head[0])

	return []float64{
		dangerFront, dangerLeft, dangerRight,
		dirUp, dirRight, dirDown, dirLeft,
		foodUp, foodDown, foodLeft, foodRight,
	}
}

func (s *Snake) Step(action int) ([]float64, float64, bool) {
	// turn
	if action == 0 {
		s.Dir = (s.Dir + 3) % 4
	} else if action == 2 {
		s.Dir = (s.Dir + 1) % 4
	}

	head := s.Body[0]
	nx, ny := nextPos(head[0], head[1], s.Dir)

	oldDist := math.Abs(float64(s.FoodX-head[0])) + math.Abs(float64(s.FoodY-head[1]))
	newDist := math.Abs(float64(s.FoodX-nx)) + math.Abs(float64(s.FoodY-ny))

	if s.danger(nx, ny) {
		return s.State(), -1.0, true
	}

	s.Body = append([][2]int{{nx, ny}}, s.Body...)

	// base step penalty
	reward := -0.01

	// 🚫 ZIG-ZAG PENALTY (KEY FIX)
	if action != 1 {
		reward -= 0.01
	}

	if nx == s.FoodX && ny == s.FoodY {
		s.Length++
		s.StepsSinceEat = 0
		reward += 1.0
		s.spawnFood()
	} else {
		if len(s.Body) > s.Length {
			s.Body = s.Body[:s.Length]
		}
		s.StepsSinceEat++
	}

	if newDist < oldDist {
		reward += 0.05
	} else if newDist > oldDist {
		reward -= 0.05
	}

	if s.StepsSinceEat > s.W*s.H {
		reward -= 1.0
		return s.State(), reward, true
	}

	return s.State(), reward, false
}

func nextPos(x, y, dir int) (int, int) {
	switch dir {
	case 0:
		return x, y - 1
	case 1:
		return x + 1, y
	case 2:
		return x, y + 1
	default:
		return x - 1, y
	}
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
